import re
import sys
import time
import logging
import hashlib
import warnings
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd


class PeptideCollapse:

    logger = logging.getLogger("PeptideCollapse")

    def __init__(self, verbose: bool = True):
        self.required_columns = {
            "essential": [
                "R.FileName",
                "EG.PrecursorId",
                "EG.TotalQuantity (Settings)",
                "PEP.PeptidePosition",
                "EG.PTMAssayProbability",
                "PG.Genes",
                "PG.ProteinGroups",
            ],
            "optional": [
                "R.Condition",
                "EG.PTMLocalizationProbabilities",
                "EG.ProteinPTMLocations",
                "PEP.StrippedSequence",
                "PG.UniProtIds",
            ],
        }

        self.data = None
        self.processed_data = None
        self.fasta_dict = None
        self.peptide_data = None
        self.site_data = None
        self.site_localization_per_run = None

        self.processing_stats = {
            'initial_rows': 0,
            'phospho_rows': 0,
            'final_peptides': 0,
            'final_sites': 0,
            'processing_time': 0,
            'per_site_localization_used': False,
            'per_site_localization_fallback_pct': 0.0,
            'noise_floor_filter': False,
            'noise_floor_removed': 0,
            'aggregation_method': '',
            'localization_strategy': '',
            'per_run_cells_masked': 0,
            'per_run_cells_total': 0,
            'sites_dropped_all_nan': 0,
            'n_samples': 0,
            'completeness_pct': 0.0,
            'localization_cutoff': 0.0,
            'sites_before_cutoff': 0,
            'sites_after_cutoff': 0,
        }

        self.verbose = verbose
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging based on verbose setting.

        Logs always go to a file (PeptideCollapse.log) in the working directory.
        When verbose=True, logs also print to console.
        """
        # Clear existing handlers to reconfigure for this instance
        self.logger.handlers.clear()

        # Always log to file
        file_handler = logging.FileHandler("PeptideCollapse.log", mode="a")
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)

        # Console only when verbose
        if self.verbose:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                "[%(levelname)s] %(name)s: %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)

        self.logger.setLevel(logging.DEBUG)

    def load_data(self, data: pd.DataFrame, validate: bool = True) -> None:

        self.data = data.copy()
        self.processing_stats['initial_rows'] = len(self.data)

        self.logger.info("Loaded %d rows", len(self.data))

        # Row count report
        self.logger.info("Total rows loaded: %d", len(self.data))

        # Sample count report
        n_samples = self.data['R.FileName'].nunique()
        self.processing_stats['n_samples'] = n_samples
        self.logger.info("Unique samples (R.FileName): %d", n_samples)

        # Duplicate raw file check
        self._check_duplicate_raw_files()

        if validate:
            self._validate_input_data()


    def _check_duplicate_raw_files(self) -> None:
        """Check if any R.FileName values map to identical data patterns."""
        file_hashes = {}
        for fname in self.data['R.FileName'].unique():
            subset = self.data.loc[self.data['R.FileName'] == fname, 'EG.PrecursorId']
            sorted_precursors = sorted(subset.dropna().astype(str).tolist())[:100]
            h = hashlib.md5("||".join(sorted_precursors).encode()).hexdigest()
            file_hashes.setdefault(h, []).append(fname)

        for h, fnames in file_hashes.items():
            if len(fnames) > 1:
                self.logger.warning(
                    "Potential file duplication detected: %s share identical "
                    "first-100 precursor patterns",
                    fnames,
                )


    def load_fasta(self, fasta_path: str) -> None:

        self.fasta_dict = self._load_fasta_to_dict(fasta_path)
        self.logger.info("Loaded FASTA with %d protein entries", len(self.fasta_dict))


    def preprocess_data(self) -> None:

        if self.data is None:
            raise ValueError("No data loaded. Use load_data() first.")

        df = self.data.copy()
        underscore_count = df["PG.Genes"].str.contains("_", na=False).sum()
        if underscore_count > 0:
            df["PG.Genes"] = df["PG.Genes"].str.replace("_", "#", regex=False)
            self.logger.warning(
                "%d gene names contained underscores and have been replaced with '#'.",
                underscore_count,
            )

        modification_info = df["EG.PrecursorId"].apply(self._extract_sequence_modifications)

        df["clean_sequence"] = [info["clean_sequence"] for info in modification_info]
        df["phospho_positions"] = [info["phospho_positions"] for info in modification_info]
        df["phospho_count"] = [info["phospho_count"] for info in modification_info]
        df["all_modifications"] = [info["all_modifications"] for info in modification_info]
        df["phospho_sequence"] = [info["phospho_sequence"] for info in modification_info]

        phospho_rows = len(df)
        df = df[df["phospho_count"] > 0].copy()
        removed_non_phospho = phospho_rows - len(df)

        self.logger.info(
            "Preprocessing: %d rows remaining after phospho filter, %d non-phospho removed",
            len(df), removed_non_phospho,
        )

        df["peptide_start_position"] = df["PEP.PeptidePosition"].astype(str).apply(
            self._extract_first_valid_position
        )

        initial_with_phospho = len(df)
        df = df.dropna(subset=["peptide_start_position"]).copy()
        df["peptide_start_position"] = df["peptide_start_position"].astype(int)

        removed_invalid_pos = initial_with_phospho - len(df)
        if removed_invalid_pos > 0:
            self.logger.warning(
                "%d rows removed due to invalid peptide start position", removed_invalid_pos
            )

        df["phospho_multiplicity"] = df["phospho_count"].apply(lambda x: min(x, 3))

        self.processed_data = df
        self.processing_stats['phospho_rows'] = len(df)

        self.logger.info("Preprocessing complete: %d phospho rows retained", len(df))


    def collapse_to_peptides(
        self,
        cutoff: float = 0.75,
        collapse_level: str = "PG",
        aggregation_method: str = "median",
        exclude_carbamidomethyl: bool = True,
        add_kinase_sequences: bool = False,
        kinase_window_size: int = 6,
        noise_floor_filter: bool = True
    ) -> pd.DataFrame:

        if self.processed_data is None:
            self.preprocess_data()

        self.logger.info("Starting peptide-level collapse (aggregation=%s, cutoff=%.2f)", aggregation_method, cutoff)

        self.peptide_data = self._create_peptide_level_collapse(
            self.processed_data, cutoff, collapse_level, aggregation_method,
            exclude_carbamidomethyl, noise_floor_filter
        )

        if add_kinase_sequences:
            if self.fasta_dict is None:
                raise ValueError("FASTA data required for kinase sequences. Use load_fasta() first.")
            self.peptide_data = self._generate_kinase_sequences(
                self.peptide_data, kinase_window_size
            )

        self.processing_stats['final_peptides'] = len(self.peptide_data)
        self.logger.info("Peptide-level collapse complete: %d peptides", len(self.peptide_data))

        return self.peptide_data

    def collapse_to_sites(
        self,
        cutoff: float = 0.75,
        collapse_level: str = "PG",
        aggregation_method: str = "median",
        add_kinase_sequences: bool = True,
        kinase_window_size: int = 6,
        noise_floor_filter: bool = True,
        localization_strategy: str = "per_run"
    ) -> pd.DataFrame:
        """Collapse precursor-level data to phosphosite-level matrix.

        Parameters
        ----------
        localization_strategy : {"per_run", "global_max"}
            "per_run" (default, bulletproof): mask the per-(site, run) intensity
            cell whenever the site's localization probability in that specific
            run is below ``cutoff``. Sites with zero remaining quant cells are
            dropped. This matches Reviewer 2's literal request — pre-filtering
            site detections per run before any wide-format quantitative
            comparison.

            "global_max" (legacy v3 / Hogrebe et al. plugin convention):
            collapse localization to a dataset-wide max per site, then apply a
            single cutoff filter. Preserves intensities in runs where the site
            is confidently localized somewhere in the dataset, even if its
            per-run loc score is borderline. Available for direct A/B
            comparison against the per_run strategy.

        Notes
        -----
        Regardless of strategy, ``self.site_localization_per_run`` is populated
        with a (PTM_Collapse_key x R.FileName) DataFrame of per-run loc probs,
        and ``count_per_run_class_I()`` will use it for per-run depth metrics.
        """

        if self.processed_data is None:
            self.preprocess_data()

        if localization_strategy not in ("per_run", "global_max"):
            raise ValueError(
                f"localization_strategy must be 'per_run' or 'global_max', "
                f"got {localization_strategy!r}"
            )

        self.logger.info(
            "Starting site-level collapse (aggregation=%s, cutoff=%.2f, strategy=%s)",
            aggregation_method, cutoff, localization_strategy,
        )

        self.site_data = self._create_site_level_collapse(
            self.processed_data, cutoff, collapse_level, aggregation_method,
            noise_floor_filter, localization_strategy
        )

        if add_kinase_sequences:
            if self.fasta_dict is None:
                raise ValueError("FASTA data required for kinase sequences. Use load_fasta() first.")
            self.site_data = self._generate_kinase_sequences(
                self.site_data, kinase_window_size
            )

        self.processing_stats['final_sites'] = len(self.site_data)
        self.logger.info("Site-level collapse complete: %d sites", len(self.site_data))
        return self.site_data

    def process_complete_pipeline(
        self,
        data: pd.DataFrame,
        cutoff: float = 0.75,
        collapse_level: str = "PG",
        aggregation_method: str = "median",
        return_both: bool = False,
        exclude_carbamidomethyl: bool = True,
        fasta_path: Optional[str] = None,
        add_kinase_sequences: bool = True,
        kinase_window_size: int = 6,
        noise_floor_filter: bool = True,
        localization_strategy: str = "per_run"
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:

        start_time = time.time()

        self.logger.info("Starting complete pipeline")

        self.load_data(data)


        if add_kinase_sequences and fasta_path:
            self.load_fasta(fasta_path)


        self.preprocess_data()

        if return_both:
            # Create both levels
            peptide_data = self.collapse_to_peptides(
                cutoff, collapse_level, aggregation_method, exclude_carbamidomethyl,
                add_kinase_sequences, kinase_window_size, noise_floor_filter
            )
            site_data = self.collapse_to_sites(
                cutoff, collapse_level, aggregation_method,
                add_kinase_sequences, kinase_window_size, noise_floor_filter,
                localization_strategy
            )

            self.processing_stats['processing_time'] = time.time() - start_time
            self.logger.info("Pipeline complete in %.2f seconds", self.processing_stats['processing_time'])
            return peptide_data, site_data
        else:
            # Site-level only
            site_data = self.collapse_to_sites(
                cutoff, collapse_level, aggregation_method,
                add_kinase_sequences, kinase_window_size, noise_floor_filter,
                localization_strategy
            )

            self.processing_stats['processing_time'] = time.time() - start_time
            self.logger.info("Pipeline complete in %.2f seconds", self.processing_stats['processing_time'])
            return site_data

    def reformat_for_analysis(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:

        if data is None:
            if self.site_data is None:
                raise ValueError("No site data available. Run collapse_to_sites() first.")
            data = self.site_data

        return self._clean_and_reformat_phospho_data(data)

    def get_processing_summary(self) -> dict:

        return self.processing_stats.copy()

    def get_quant_sample_data(self) -> list:

        return self.data['R.FileName'].unique().tolist()

    def get_precursor_condition_dataset(self) -> pd.DataFrame:
        return pd.DataFrame({'Sample': self.get_quant_sample_data(), 'Condition': np.nan})

    def calculate_selectivity(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate per-sample phosphopeptide enrichment selectivity from raw data.

        Returns a DataFrame with columns: Sample, total_precursors,
        phospho_precursors, selectivity_pct.
        """
        if data is not None:
            self.load_data(data, validate=False)
        elif self.data is None:
            raise ValueError("No data loaded. Pass a DataFrame or use load_data() first.")

        df = self.data[['R.FileName', 'EG.PrecursorId']].drop_duplicates()
        df = df.copy()
        df['is_phospho'] = df['EG.PrecursorId'].str.contains(
            r'\[Phospho \(STY\)\]', regex=True, na=False
        )

        summary = (
            df.groupby('R.FileName')
            .agg(
                total_precursors=('EG.PrecursorId', 'count'),
                phospho_precursors=('is_phospho', 'sum'),
            )
            .reset_index()
            .rename(columns={'R.FileName': 'Sample'})
        )
        summary['selectivity_pct'] = (
            summary['phospho_precursors'] / summary['total_precursors'] * 100
        ).round(2)
        return summary

    def validate_output(self, data: Optional[pd.DataFrame] = None) -> dict:
        """Validate the output DataFrame for data integrity issues.

        Parameters
        ----------
        data : pd.DataFrame, optional
            The collapsed output to validate.  Falls back to self.site_data.

        Returns
        -------
        dict with validation results.
        """
        if data is None:
            data = self.site_data
        if data is None:
            raise ValueError("No data to validate. Run a collapse method first or pass a DataFrame.")

        issues = []

        # Determine sample columns (everything that is not metadata)
        metadata_cols = {
            "UPD_seq", "PTM_localization", "Protein_group", "Gene_group",
            "PTM_Collapse_key", "kinase_sequence", "Protein_Collapse_key",
            "PG.Genes", "PG.ProteinGroups", "clean_sequence", "all_modifications",
            "phospho_count", "EG.PTMAssayProbability", "peptide_collapse_key",
        }
        sample_cols = [c for c in data.columns if c not in metadata_cols]

        # 1. No -inf values in sample columns
        numeric_sample = data[sample_cols].select_dtypes(include=[np.number])
        neg_inf_count = (numeric_sample == -np.inf).sum().sum()
        if neg_inf_count > 0:
            msg = f"Found {neg_inf_count} -inf values in sample columns"
            issues.append(msg)
            self.logger.error(msg)
        else:
            self.logger.info("No -inf values in sample columns")

        # 2. No duplicate PTM_Collapse_key values
        if "PTM_Collapse_key" in data.columns:
            dup_count = data["PTM_Collapse_key"].duplicated().sum()
            if dup_count > 0:
                msg = f"Found {dup_count} duplicate PTM_Collapse_key values"
                issues.append(msg)
                self.logger.error(msg)
            else:
                self.logger.info("No duplicate PTM_Collapse_key values")

            # 3. PTM_Collapse_key format check: ProteinGroup~Gene_AAsitepos_Mmult
            pattern = re.compile(r'^[^~]+~[^_]+_[A-Za-z]\d+_M\d+$')
            bad_keys = data["PTM_Collapse_key"].apply(lambda k: not bool(pattern.match(str(k))))
            bad_count = bad_keys.sum()
            if bad_count > 0:
                examples = data.loc[bad_keys, "PTM_Collapse_key"].head(5).tolist()
                msg = f"{bad_count} PTM_Collapse_key values don't match expected format. Examples: {examples}"
                issues.append(msg)
                self.logger.warning(msg)
            else:
                self.logger.info("All PTM_Collapse_key values match expected format")

            # 4. PTM_0_aa in collapse key must be S, T, or Y
            def _extract_aa(key):
                try:
                    ptm_part = str(key).split("~")[1]
                    site_part = ptm_part.split("_")[1]
                    return site_part[0]
                except (IndexError, TypeError):
                    return "?"

            aa_values = data["PTM_Collapse_key"].apply(_extract_aa)
            invalid_aa = aa_values[~aa_values.isin(["S", "T", "Y"])]
            if len(invalid_aa) > 0:
                msg = f"{len(invalid_aa)} sites have non-STY amino acid in collapse key"
                issues.append(msg)
                self.logger.warning(msg)
            else:
                self.logger.info("All site amino acids are S, T, or Y")

        # 5. Sample values in reasonable log2 range (0-35) or NaN
        out_of_range = 0
        for col in numeric_sample.columns:
            vals = numeric_sample[col].dropna()
            oor = ((vals < 0) | (vals > 35)).sum()
            out_of_range += oor
        if out_of_range > 0:
            msg = f"{out_of_range} sample values outside expected log2 range [0, 35]"
            issues.append(msg)
            self.logger.warning(msg)
        else:
            self.logger.info("All sample values within expected log2 range [0, 35]")

        # 6. Completeness percentage
        total_cells = numeric_sample.size
        nan_cells = numeric_sample.isna().sum().sum()
        completeness = ((total_cells - nan_cells) / total_cells * 100) if total_cells > 0 else 0.0
        self.logger.info("Output completeness: %.1f%% (%d/%d cells)", completeness, total_cells - nan_cells, total_cells)

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "neg_inf_count": neg_inf_count,
            "out_of_range_count": out_of_range,
            "completeness_pct": round(completeness, 2),
            "n_sites": len(data),
            "n_sample_cols": len(numeric_sample.columns),
        }

    def count_per_run_class_I(self, cutoff: float = 0.75) -> pd.Series:
        """Count per-run Class I localized phosphosites.

        A site is Class I in a given run when (a) its quantitative value is
        non-NaN in that run AND (b) its per-(site, run) localization probability
        is >= cutoff. This is the per-run depth metric Reviewer 2 asked for —
        it answers "how many sites are confidently localized in *this specific
        run*?" rather than "how many sites are confidently localized somewhere
        in the dataset and detected in this run?"

        Works regardless of which ``localization_strategy`` was used to produce
        ``site_data``: it always cross-references the quant matrix against
        ``self.site_localization_per_run`` per cell.

        Returns
        -------
        pd.Series indexed by R.FileName, values are per-run Class I site counts.
        """
        if self.site_data is None:
            raise ValueError("No site data. Run collapse_to_sites() first.")
        if self.site_localization_per_run is None:
            raise ValueError(
                "Per-run localization matrix not available. Re-run collapse_to_sites()."
            )

        quant = self.site_data.set_index("PTM_Collapse_key", drop=False)
        metadata = {
            "UPD_seq", "PTM_localization", "Protein_group", "Gene_group",
            "PTM_Collapse_key", "kinase_sequence", "Protein_Collapse_key",
            "PG.Genes", "PG.ProteinGroups",
        }
        sample_cols = [
            c for c in quant.columns
            if c not in metadata and pd.api.types.is_numeric_dtype(quant[c])
        ]
        quant_samples = quant[sample_cols]

        loc = self.site_localization_per_run.reindex(
            index=quant_samples.index, columns=sample_cols
        )

        class_I = quant_samples.notna() & (loc >= cutoff)
        return class_I.sum(axis=0).rename("class_I_count")

    # Private methods (implementation of original functions)

    def _validate_input_data(self) -> None:

        missing_columns = []
        present_columns = []
        required_columns = self.required_columns['essential']

        for column in required_columns:
            if column in self.data.columns:
                present_columns.append(column)
            else:
                missing_columns.append(column)
                self.logger.error("Column missing: %s", column)

        if missing_columns:
            self.logger.error(
                "Missing %d required column(s): %s", len(missing_columns), missing_columns
            )
            self.logger.debug(
                "Available columns in dataset: %s", sorted(self.data.columns.tolist())
            )
            sys.exit("Function stopped due to missing required columns")

    def _extract_sequence_modifications(self, sequence: str) -> Dict[str, Union[List[int], str, int]]:

        base_sequence = sequence

        if len(sequence) >= 4:
            if sequence.startswith("*.") or sequence.startswith("_."):
                base_sequence = base_sequence[2:]
            elif sequence.startswith("*") or sequence.startswith("_"):
                base_sequence = base_sequence[1:]

            if ".*" in base_sequence:
                base_sequence = base_sequence.split(".*")[0]
            elif "._" in base_sequence:
                base_sequence = base_sequence.split("._")[0]
            elif base_sequence.endswith("*"):
                base_sequence = base_sequence[:-1]
            elif base_sequence.endswith("_"):
                base_sequence = base_sequence[:-1]

            parts = base_sequence.split(".")
            if len(parts) > 1 and parts[-1].isdigit():
                base_sequence = ".".join(parts[:-1])

        pat_del_all = r"\[(?!Phospho \(STY\))[^\]]*\]"
        phospho_only_sequence = re.sub(pat_del_all, "", base_sequence)
        clean_sequence = re.sub(r"\[[^\]]*\]", "", base_sequence)
        clean_sequence = clean_sequence.replace("_", "")

        phospho_count = len(sequence.split("[Phospho (STY)]")) - 1
        all_modifications = re.findall(r"\[([^\]]+)\]", base_sequence)

        phospho_positions = []
        if phospho_count > 0:
            phospho_positions = self._calculate_phospho_positions(phospho_only_sequence)

        return {
            "clean_sequence": clean_sequence,
            "phospho_positions": phospho_positions,
            "phospho_count": phospho_count,
            "all_modifications": all_modifications,
            "phospho_sequence": phospho_only_sequence,
            "base_sequence": base_sequence,
        }

    def _calculate_phospho_positions(self, phospho_sequence: str) -> List[int]:

        if "[Phospho (STY)]" not in phospho_sequence:
            return []

        segments = phospho_sequence.split("[Phospho (STY)]")
        positions = []
        current_pos = 0

        for i in range(len(segments) - 1):
            current_pos += len(segments[i])
            positions.append(current_pos)
        return positions

    def _extract_first_valid_position(self, position_str: str) -> Optional[int]:

        if pd.isna(position_str) or position_str == "None" or position_str == "":
            return None

        try:
            parts = str(position_str).split(";")
            if parts:
                first_part = parts[0].split(",")[0].strip()
                if first_part and first_part != "None":
                    return int(first_part)
        except (ValueError, AttributeError):
            pass
        return None

    def _create_peptide_level_collapse(
        self, data: pd.DataFrame, cutoff: float, collapse_level: str,
        aggregation_method: str, exclude_carbamidomethyl: bool,
        noise_floor_filter: bool = True
    ) -> pd.DataFrame:
        df = data.copy()

        # Re-extract modifications
        modification_info = df["EG.PrecursorId"].apply(self._extract_sequence_modifications)
        df["clean_sequence"] = [info["clean_sequence"] for info in modification_info]
        df["phospho_positions"] = [info["phospho_positions"] for info in modification_info]
        df["phospho_count"] = [info["phospho_count"] for info in modification_info]
        df["all_modifications"] = [info["all_modifications"] for info in modification_info]

        rows_before_phospho_filter = len(df)
        df = df[df["phospho_count"] > 0].copy()
        removed_non_phospho = rows_before_phospho_filter - len(df)

        self.logger.info(
            "Peptide collapse preprocessing: %d rows remaining, %d phospho, %d non-phospho removed",
            len(df), len(df), removed_non_phospho,
        )

        if len(df) == 0:
            self.logger.warning("No phospho rows remaining after filter; returning empty DataFrame")
            return pd.DataFrame()

        df["genes_processed"] = df["PG.Genes"].astype(str).str.replace("#", "_", regex=False)
        df["peptide_collapse_key"] = df.apply(
            lambda row: self._create_peptide_key(row, collapse_level, exclude_carbamidomethyl), axis=1
        )

        error_keys = df["peptide_collapse_key"].str.startswith("Error_")
        if error_keys.any():
            self.logger.warning("%d rows with error collapse keys removed", error_keys.sum())
            df = df[~error_keys].copy()

        if len(df) == 0:
            self.logger.warning("No rows remaining after error key removal; returning empty DataFrame")
            return pd.DataFrame()

        quant_pivot = df.pivot_table(
            index="peptide_collapse_key",
            columns="R.FileName",
            values="EG.TotalQuantity (Settings)",
            aggfunc="sum",
        )
        quant_pivot = quant_pivot.replace(0, np.nan)

        self.logger.info(
            "Peptide pivot: %d keys x %d samples, NaN count: %d, zero count: %d",
            quant_pivot.shape[0], quant_pivot.shape[1],
            int(quant_pivot.isna().sum().sum()),
            int((quant_pivot == 0).sum().sum()),
        )

        loc_pivot = df.pivot_table(
            index="peptide_collapse_key",
            columns="R.FileName",
            values="EG.PTMAssayProbability",
            aggfunc="first",
        )
        loc_pivot = loc_pivot.fillna(-1)
        loc_pivot = pd.DataFrame({"EG.PTMAssayProbability": loc_pivot.apply(max, axis=1)})

        metadata_cols = ["PG.Genes", "PG.ProteinGroups", "clean_sequence", "all_modifications", "phospho_count"]
        available_metadata_cols = [col for col in metadata_cols if col in df.columns]

        if available_metadata_cols:
            metadata_pivot = df.pivot_table(
                index="peptide_collapse_key", values=available_metadata_cols, aggfunc="first"
            )
        else:
            metadata_pivot = pd.DataFrame(index=quant_pivot.index)

        _pept_sample_cols = [c for c in quant_pivot.columns if c not in ("peptide_collapse_key",)]

        self.processing_stats['aggregation_method'] = aggregation_method

        if aggregation_method == "consolidate":
            _pept_grouped = quant_pivot.reset_index().groupby("peptide_collapse_key")[_pept_sample_cols]
            quant_final = _pept_grouped.apply(
                lambda g: pd.Series(
                    self._consolidate(g.values),
                    index=_pept_sample_cols,
                )
            )
        elif aggregation_method == "median":
            quant_final = quant_pivot.groupby(level=0).median()
        elif aggregation_method == "mean":
            quant_final = quant_pivot.groupby(level=0).mean()
        else:
            quant_final = quant_pivot.groupby(level=0).sum()

        self.logger.info(
            "Peptide aggregation: %d collapse keys, NaN count in aggregated matrix: %d",
            len(quant_final), int(quant_final.isna().sum().sum()),
        )

        # Log2 transform after consolidation (matching OG pipeline)
        quant_final = np.log2(quant_final.replace(0, np.nan))

        neg_inf_count = int((quant_final == -np.inf).sum().sum())
        if neg_inf_count > 0:
            self.logger.warning("Log2 produced %d -inf values", neg_inf_count)
        else:
            self.logger.debug("Log2 transform: no -inf values produced")

        self.processing_stats['noise_floor_filter'] = noise_floor_filter
        if noise_floor_filter:
            before_nf = int(quant_final.notna().sum().sum())
            quant_final = quant_final.replace(0, np.nan).replace(1, np.nan)
            after_nf = int(quant_final.notna().sum().sum())
            nf_removed = before_nf - after_nf
            self.processing_stats['noise_floor_removed'] = self.processing_stats.get('noise_floor_removed', 0) + nf_removed
            self.logger.info("Noise floor filter removed %d values", nf_removed)

        quant_final = quant_final.replace("Filtered", np.nan)
        loc_final = loc_pivot.groupby(level=0).max()

        result = pd.concat([quant_final, metadata_pivot, loc_final], axis=1)
        result = result.reset_index()

        result["EG.PTMAssayProbability"] = result["EG.PTMAssayProbability"].replace(-1, np.nan)
        pre_filter_count = len(result)
        result = result[result["EG.PTMAssayProbability"] >= cutoff]

        self.logger.info(
            "Peptide localization cutoff (%.2f): %d before, %d after, %d removed",
            cutoff, pre_filter_count, len(result), pre_filter_count - len(result),
        )

        new_columns = []
        for col in result.columns:
            if isinstance(col, tuple):
                new_columns.append(col[1] if len(col) > 1 else col[0])
            else:
                new_columns.append(col)
        result.columns = new_columns

        if "PG.Genes" in result.columns:
            result["PG.Genes"] = result["PG.Genes"].astype(str).str.replace("#", "_", regex=False)

        # Final output summary
        sample_result_cols = [c for c in result.columns if c not in metadata_cols and c != "peptide_collapse_key" and c != "EG.PTMAssayProbability"]
        numeric_result = result[sample_result_cols].select_dtypes(include=[np.number])
        total_cells = numeric_result.size
        nan_cells = int(numeric_result.isna().sum().sum())
        completeness = ((total_cells - nan_cells) / total_cells * 100) if total_cells > 0 else 0.0
        self.logger.info(
            "Peptide final output: %d peptides, %d samples, completeness %.1f%%, NaN %.1f%%",
            len(result), len(numeric_result.columns), completeness, 100.0 - completeness,
        )

        return result

    def _create_site_level_collapse(
        self, data: pd.DataFrame, cutoff: float, collapse_level: str,
        aggregation_method: str, noise_floor_filter: bool = True,
        localization_strategy: str = "per_run"
    ) -> pd.DataFrame:
        """Create site-level collapsed data."""
        df = data.copy()

        modification_info = df["EG.PrecursorId"].apply(self._extract_sequence_modifications)
        df["PTM_base_seq"] = [info["clean_sequence"] for info in modification_info]
        df["PTM_0_pos_val"] = [info["phospho_positions"] for info in modification_info]
        df["PTM_0_num"] = [info["phospho_count"] for info in modification_info]
        df["PTM_group"] = df["EG.PrecursorId"]

        rows_before = len(df)
        df = df[df["PTM_0_num"] > 0]
        removed_non_phospho = rows_before - len(df)

        self.logger.info(
            "Site collapse preprocessing: %d rows remaining, %d phospho, %d non-phospho removed",
            len(df), len(df), removed_non_phospho,
        )

        # Per-site localization: parse EG.PTMLocalizationProbabilities for per-site
        # probabilities. Positions are kept from EG.PrecursorId to ensure consistency
        # across charge states and avoid data fragmentation across nearby sites.
        _use_loc_string = "EG.PTMLocalizationProbabilities" in df.columns
        self.processing_stats['per_site_localization_used'] = _use_loc_string

        if _use_loc_string:
            _loc_dicts = df["EG.PTMLocalizationProbabilities"].apply(
                self._parse_localization_probabilities
            )
            df["_loc_dict"] = _loc_dicts.values

        df = df.explode("PTM_0_pos_val")

        df["UPD_seq"] = df.apply(
            lambda x: self._create_modified_sequence(x["PTM_base_seq"], x["PTM_0_pos_val"]), axis=1
        )

        df = df.reset_index()
        df["PTM_0_aa"] = df.apply(
            lambda x: self._get_phospho_amino_acid(x["PTM_base_seq"], x["PTM_0_pos_val"]), axis=1
        )
        df = df.set_index("index")

        if _use_loc_string:
            # Look up per-site probability from the stored parsed dict
            df["PTM_localization"] = [
                d.get(int(pos), np.nan) if d else np.nan
                for d, pos in zip(df["_loc_dict"], df["PTM_0_pos_val"])
            ]
            df = df.drop(columns=["_loc_dict"])
            # Fall back to joint probability where per-site parsing failed
            _fallback = df["PTM_localization"].isna()
            df.loc[_fallback, "PTM_localization"] = (
                df.loc[_fallback, "EG.PTMAssayProbability"].astype(np.float64)
            )

            n_per_site = int((~_fallback).sum())
            n_fallback = int(_fallback.sum())
            total_loc = n_per_site + n_fallback
            fallback_pct = (n_fallback / total_loc * 100) if total_loc > 0 else 0.0
            self.processing_stats['per_site_localization_fallback_pct'] = round(fallback_pct, 2)

            self.logger.info(
                "Per-site localization: %d rows got per-site prob, %d fell back to joint prob (%.1f%% fallback)",
                n_per_site, n_fallback, fallback_pct,
            )
        else:
            df["PTM_localization"] = df["EG.PTMAssayProbability"].astype(np.float64)
            self.processing_stats['per_site_localization_fallback_pct'] = 100.0
            self.logger.info("Per-site localization column not available; using joint EG.PTMAssayProbability for all rows")

        fine_names = list(data["R.FileName"].unique())
        self.processing_stats['n_samples'] = len(fine_names)

        df2 = pd.pivot_table(
            df, index=["PTM_group", "PTM_0_pos_val"], columns=["R.FileName"],
            values=["EG.TotalQuantity (Settings)"], aggfunc="sum"
        )
        df2 = df2.replace(0, np.nan)

        self.logger.info(
            "Site pivot: %d keys x %d samples, NaN count: %d, zero count: %d",
            df2.shape[0], df2.shape[1],
            int(df2.isna().sum().sum()),
            int((df2 == 0).sum().sum()),
        )

        df3 = pd.pivot_table(
            df, index=["PTM_group", "PTM_0_pos_val"], columns=["R.FileName"],
            values=["PTM_localization"], aggfunc="first"
        )
        df3 = df3.fillna(-1)
        df3 = pd.DataFrame({"PTM_localization": df3.apply(max, axis=1)})

        keep = ["PEP.PeptidePosition", "PG.ProteinGroups", "PG.Genes", "PTM_0_num", "PTM_0_aa"]
        df4 = pd.pivot_table(df, index=["PTM_group", "PTM_0_pos_val"], values=keep, aggfunc="first")

        df5 = pd.pivot_table(
            df, index=["PTM_group", "PTM_0_pos_val"], values=["UPD_seq"], aggfunc="first"
        )

        data_combined = pd.concat([df2, df4, df3, df5], axis=1).reset_index()

        data_combined["PTM_Genprot"] = data_combined["PG.Genes"].astype(str)
        data_combined["PTM_pep_pos"] = (
            data_combined["PEP.PeptidePosition"].astype(str)
            .apply(lambda row: list(filter(None, row.split(";")))[0])
        )
        data_combined["PTM_pep_pos"] = (
            data_combined["PTM_pep_pos"].astype(str)
            .apply(lambda row: list(filter(None, row.split(",")))[0])
        )
        data_combined = data_combined[data_combined["PTM_pep_pos"] != "None"]
        data_combined["PTM_pep_pos"] = data_combined["PTM_pep_pos"].astype(np.int64)

        data_combined["PTM_mult123"] = data_combined["PTM_0_num"].astype(np.int64)
        data_combined["PTM_mult123"] = data_combined["PTM_mult123"].apply(lambda x: min(x, 3))

        data_combined["PTM_Collapse_key"] = data_combined.apply(
            lambda x: self._create_collapse_key(
                x["PG.ProteinGroups"], x["PTM_Genprot"], x["PTM_0_aa"],
                x["PTM_0_pos_val"], x["PTM_pep_pos"], x["PTM_mult123"]
            ), axis=1
        )

        # Per-(site, run) localization matrix at PTM_Collapse_key resolution.
        # For each (site, run) cell we take the strongest per-precursor loc prob
        # observed for that site in that run — i.e. evidence from any precursor
        # mapping to the site is sufficient to localize it in that run.
        _key_map = data_combined[["PTM_group", "PTM_0_pos_val", "PTM_Collapse_key"]].drop_duplicates()
        _df_keyed = df.reset_index(drop=False).merge(
            _key_map, on=["PTM_group", "PTM_0_pos_val"], how="inner"
        )
        loc_per_run = _df_keyed.pivot_table(
            index="PTM_Collapse_key",
            columns="R.FileName",
            values="PTM_localization",
            aggfunc="max",
        )
        # Reorder columns to match the canonical sample order downstream
        loc_per_run = loc_per_run.reindex(columns=fine_names)
        self.site_localization_per_run = loc_per_run
        self.logger.info(
            "Per-(site, run) localization matrix built: %d sites x %d runs (%.1f%% non-NaN)",
            loc_per_run.shape[0], loc_per_run.shape[1],
            100.0 * loc_per_run.notna().sum().sum() / loc_per_run.size if loc_per_run.size > 0 else 0,
        )

        cols = []
        for c in fine_names:
            cols.append([col for col in data_combined.columns if c in col][0])

        df1 = data_combined.set_index("PTM_Collapse_key")
        df2_agg = df1[df1.columns[df1.columns.isin(cols)]].reset_index()

        sample_cols = [c for c in df2_agg.columns if c != "PTM_Collapse_key"]

        self.processing_stats['aggregation_method'] = aggregation_method

        if aggregation_method == "consolidate":
            # OG-style ratio-based imputation + sum (in linear intensity space)
            groups = df2_agg.groupby("PTM_Collapse_key")[sample_cols]
            consolidated = groups.apply(
                lambda g: pd.Series(
                    self._consolidate(g.values),
                    index=sample_cols,
                )
            )
            df3_agg = consolidated
        elif aggregation_method == "median":
            df3_agg = df2_agg.groupby("PTM_Collapse_key")[sample_cols].median()
        elif aggregation_method == "mean":
            df3_agg = df2_agg.groupby("PTM_Collapse_key")[sample_cols].mean()
        else:
            df3_agg = df2_agg.groupby("PTM_Collapse_key")[sample_cols].sum()

        self.logger.info(
            "Site aggregation: %d collapse keys, NaN count in aggregated matrix: %d",
            len(df3_agg), int(df3_agg.isna().sum().sum()),
        )

        # Per-(site, run) localization mask. Applied in linear space before log2
        # so masked cells become NaN naturally through the log2 step. Mask granularity
        # is (PTM_Collapse_key x R.FileName) — i.e. for each site, each run is judged
        # independently against the cutoff. NaN per-run loc is treated as below cutoff.
        self.processing_stats['localization_strategy'] = localization_strategy
        if localization_strategy == "per_run":
            # df3_agg columns are tuples like ('EG.TotalQuantity (Settings)', '<filename>')
            # — pull the filename for alignment with loc_per_run, then restore.
            _fname_for_col = {
                c: (c[1] if isinstance(c, tuple) and len(c) > 1 else c)
                for c in df3_agg.columns
            }
            _ordered_fnames = [_fname_for_col[c] for c in df3_agg.columns]
            _loc_aligned = loc_per_run.reindex(
                index=df3_agg.index, columns=_ordered_fnames
            )
            _loc_aligned.columns = list(df3_agg.columns)
            # mask=True -> keep, mask=False -> NaN. NaN loc compares False, so masked.
            _keep_mask = _loc_aligned >= cutoff
            n_pre = int(df3_agg.notna().sum().sum())
            df3_agg = df3_agg.where(_keep_mask, np.nan)
            n_post = int(df3_agg.notna().sum().sum())
            n_masked = n_pre - n_post
            self.processing_stats['per_run_cells_masked'] = n_masked
            self.processing_stats['per_run_cells_total'] = n_pre
            self.logger.info(
                "Per-run localization mask (cutoff=%.2f): %d/%d quant cells masked (%.1f%%)",
                cutoff, n_masked, n_pre,
                100.0 * n_masked / n_pre if n_pre > 0 else 0.0,
            )

        # Log2 transform after consolidation (matching OG pipeline)
        df3_agg = np.log2(df3_agg.replace(0, np.nan))

        neg_inf_count = int((df3_agg == -np.inf).sum().sum())
        if neg_inf_count > 0:
            self.logger.warning("Log2 produced %d -inf values", neg_inf_count)
        else:
            self.logger.debug("Log2 transform: no -inf values produced")

        self.processing_stats['noise_floor_filter'] = noise_floor_filter
        if noise_floor_filter:
            before_nf = int(df3_agg.notna().sum().sum())
            df3_agg = df3_agg.replace(0, np.nan).replace(1, np.nan)
            after_nf = int(df3_agg.notna().sum().sum())
            nf_removed = before_nf - after_nf
            self.processing_stats['noise_floor_removed'] = self.processing_stats.get('noise_floor_removed', 0) + nf_removed
            self.logger.info("Noise floor filter removed %d values", nf_removed)

        df3_agg = df3_agg.replace("Filtered", np.nan)

        df4_agg = df1[df1.columns[df1.columns == "PTM_localization"]].reset_index()
        df5_agg = df4_agg.groupby("PTM_Collapse_key").max()

        protgen = ["PG.Genes", "PG.ProteinGroups", "UPD_seq"]
        df6_agg = df1[df1.columns[df1.columns.isin(protgen)]].reset_index()
        df7_agg = df6_agg.groupby("PTM_Collapse_key").first()

        result = pd.concat([df3_agg, df7_agg, df5_agg], axis=1).reset_index()

        result["PTM_localization"] = result["PTM_localization"].replace(-1, np.nan)

        sites_before_cutoff = len(result)
        self.processing_stats['localization_cutoff'] = cutoff
        self.processing_stats['sites_before_cutoff'] = sites_before_cutoff

        if localization_strategy == "per_run":
            # Per-run masking already removed all unconfident (site, run) cells.
            # A site survives iff at least one run kept its quant value — i.e.
            # the row is not entirely NaN across sample columns. This replaces
            # the legacy global cross-file max filter.
            _result_sample_cols = [
                c for c in result.columns
                if c not in ("PTM_Collapse_key", "PTM_localization", "PG.Genes",
                             "PG.ProteinGroups", "UPD_seq")
            ]
            _quant_block = result[_result_sample_cols].select_dtypes(include=[np.number])
            _all_nan = _quant_block.isna().all(axis=1) if _quant_block.shape[1] > 0 else pd.Series(False, index=result.index)
            n_drop = int(_all_nan.sum())
            self.processing_stats['sites_dropped_all_nan'] = n_drop
            result = result[~_all_nan]
            self.logger.info(
                "Per-run mode: dropped %d sites with no remaining quant after masking",
                n_drop,
            )
        else:
            # Legacy "global_max" strategy: filter on the dataset-wide maximum
            # localization probability (Hogrebe et al. plugin convention).
            result = result[result["PTM_localization"] >= cutoff]

        sites_after_cutoff = len(result)
        self.processing_stats['sites_after_cutoff'] = sites_after_cutoff

        self.logger.info(
            "Site filtering (strategy=%s, cutoff=%.2f): %d before, %d after, %d removed",
            localization_strategy, cutoff, sites_before_cutoff, sites_after_cutoff,
            sites_before_cutoff - sites_after_cutoff,
        )

        cols = []
        for c in list(result.columns):
            if type(c) == tuple:
                cols.append(c[1])
            else:
                cols.append(c)
        result.columns = cols

        final_result = self._finalize_collapsed_output(result, collapse_level)

        # Final output summary
        _meta_set = {"UPD_seq", "PTM_localization", "Protein_group", "Gene_group",
                      "PTM_Collapse_key", "kinase_sequence", "Protein_Collapse_key",
                      "PG.Genes", "PG.ProteinGroups"}
        _sample_cols_final = [c for c in final_result.columns if c not in _meta_set]
        _numeric_final = final_result[_sample_cols_final].select_dtypes(include=[np.number])
        _total_cells = _numeric_final.size
        _nan_cells = int(_numeric_final.isna().sum().sum())
        _completeness = ((_total_cells - _nan_cells) / _total_cells * 100) if _total_cells > 0 else 0.0
        self.processing_stats['completeness_pct'] = round(_completeness, 2)

        self.logger.info(
            "Site final output: %d sites, %d samples, completeness %.1f%%, NaN %.1f%%",
            len(final_result), len(_numeric_final.columns), _completeness, 100.0 - _completeness,
        )

        return final_result

    def _create_peptide_key(self, row: pd.Series, collapse_level: str, exclude_carbamidomethyl: bool) -> str:

        try:
            protein_groups = row["PG.ProteinGroups"]
            if pd.isna(protein_groups):
                protein_part = "Unknown"
            else:
                protein_groups_str = str(protein_groups)
                if collapse_level == "PG":
                    protein_part = protein_groups_str.split(";")[0]
                else:
                    protein_part = protein_groups_str

            genes = row["PG.Genes"]
            if pd.isna(genes):
                gene_part = "Unknown"
            else:
                genes_str = str(genes).replace("#", "_")
                gene_part = genes_str.split(";")[0]

            clean_seq = row["clean_sequence"]
            sequence_part = str(clean_seq) if pd.notna(clean_seq) else "Unknown"

            peptide_pos = row["peptide_start_position"]
            position_part = str(int(peptide_pos)) if pd.notna(peptide_pos) else "0"

            all_mods = row["all_modifications"]
            try:
                if all_mods is None or (isinstance(all_mods, float) and np.isnan(all_mods)):
                    mod_part = "NoMods"
                elif isinstance(all_mods, (list, tuple, np.ndarray)):
                    if len(all_mods) > 0:
                        filtered_mods = all_mods
                        if exclude_carbamidomethyl:
                            filtered_mods = [
                                mod for mod in all_mods
                                if not (isinstance(mod, str) and "Carbamidomethyl" in mod)
                            ]

                        if len(filtered_mods) > 0:
                            mod_part = "_".join(sorted([str(mod) for mod in filtered_mods]))
                        else:
                            mod_part = "NoMods"
                    else:
                        mod_part = "NoMods"
                else:
                    single_mod = str(all_mods) if all_mods else ""
                    if exclude_carbamidomethyl and "Carbamidomethyl" in single_mod:
                        mod_part = "NoMods"
                    else:
                        mod_part = single_mod if single_mod else "NoMods"
            except (TypeError, AttributeError, ValueError):
                mod_part = "NoMods"

            key = f"{protein_part}~{gene_part}_{sequence_part}_{position_part}_{mod_part}"
            return key

        except Exception as e:
            return "Error_peptide_key"

    def _create_collapse_key(self, entry0, entry1, entry2, entry3, entry4, entry5) -> str:

        try:
            absolute_position = int(entry3 + entry4 - 1)
            result = (
                str(entry0) + "~" + str(entry1) + "_" + str(entry2) +
                str(absolute_position) + "_M" + str(int(entry5))
            )
            return result
        except Exception:
            return "Error_key"

    def _create_modified_sequence(self, clean_sequence: str, phospho_position: int) -> str:

        if phospho_position < 1 or phospho_position > len(clean_sequence):
            return clean_sequence

        pos_idx = phospho_position - 1
        modified_seq = (
            clean_sequence[:pos_idx] + clean_sequence[pos_idx].lower() + "*" +
            clean_sequence[pos_idx + 1:]
        )
        return modified_seq

    def _get_phospho_amino_acid(self, sequence: str, position: int) -> str:

        try:
            if position < 1 or position > len(sequence):
                return "X"
            result = sequence[position - 1 : position]
            return result if result else "X"
        except (IndexError, TypeError):
            return "X"

    @staticmethod
    def _consolidate(cons: np.ndarray) -> np.ndarray:
        """Ratio-based missing value imputation + sum, matching the OG R consolidate().

        Input: 2D array (rows=precursors, cols=samples) in LINEAR intensity space.
        Returns: 1D array (one value per sample) = sum of imputed rows.

        Algorithm:
        1. Sort rows by median intensity (ascending).
        2. While any sample has NaN among rows that have some signal:
           - For each row with NaN values, estimate them using inter-row ratios:
             for each other row, compute median(this_row / other_row) across shared
             non-NaN samples, then predict missing = ratio * other_row_value.
             Take median across predictions from all other rows.
           - If no progress (matrix unchanged), drop the row with the most NaN.
        3. Sum all remaining rows per sample.
        """
        if cons.shape[0] == 0:
            return np.full(cons.shape[1], np.nan)

        # Pre-filter: remove rows with <=1 non-NaN value (matching OG line 1311)
        non_na_per_row = np.sum(~np.isnan(cons), axis=1)
        cons = cons[non_na_per_row > 1].copy()

        if cons.shape[0] == 0:
            return np.full(cons.shape[1], np.nan)
        if cons.shape[0] == 1:
            return cons[0].copy()

        # Sort by row median (lowest first, matching OG line 1234)
        row_medians = np.nanmedian(cons, axis=1)
        cons = cons[np.argsort(row_medians)].copy()

        max_iter = cons.shape[0] * 10  # safety limit
        iteration = 0
        while iteration < max_iter:
            # Check if any sample still has NaN among rows with at least some signal
            row_has_signal = np.any(~np.isnan(cons), axis=1)
            if not np.any(row_has_signal):
                break
            active = cons[row_has_signal]
            if not np.any(np.isnan(active)):
                break

            old_cons = cons.copy()
            n_rows = cons.shape[0]

            for r in range(n_rows):
                row = cons[r]
                na_mask = np.isnan(row)
                if not np.any(na_mask) or np.all(na_mask):
                    continue

                # Estimate missing values from other rows
                predictions = []
                for other_r in range(n_rows):
                    if other_r == r:
                        continue
                    other_row = cons[other_r]
                    # Shared non-NaN positions
                    shared = ~np.isnan(row) & ~np.isnan(other_row)
                    if not np.any(shared):
                        continue
                    # Median ratio: this_row / other_row across shared samples
                    ratios = row[shared] / other_row[shared]
                    ratios = ratios[np.isfinite(ratios)]
                    if len(ratios) == 0:
                        continue
                    med_ratio = np.nanmedian(ratios)
                    if not np.isfinite(med_ratio) or med_ratio == 0:
                        continue
                    # Predict missing positions: ratio * other_row
                    pred = med_ratio * other_row[na_mask]
                    predictions.append(pred)

                if predictions:
                    pred_matrix = np.array(predictions)
                    imputed = np.nanmedian(pred_matrix, axis=0)
                    cons[r, na_mask] = imputed

            # Check if we made progress (OG: identical(cons, tempc) — checks values)
            if np.array_equal(old_cons, cons, equal_nan=True):
                # No progress — drop row with most NaN (= fewest non-NaN, matching OG line 1271)
                na_counts = np.sum(np.isnan(cons), axis=1)
                worst_row = np.argmax(na_counts)
                if na_counts[worst_row] == 0:
                    break
                cons = np.delete(cons, worst_row, axis=0)
                if cons.shape[0] == 0:
                    return np.full(old_cons.shape[1], np.nan)

            iteration += 1

        # Sum all rows per sample (matching OG line 1288)
        result = np.nansum(cons, axis=0)
        # Restore NaN for samples where ALL rows were NaN
        all_nan = np.all(np.isnan(cons), axis=0)
        result[all_nan] = np.nan
        return result

    def _rank_select_positions(
        self, loc_dict: Dict[int, float], n: int
    ) -> Tuple[List[int], List[float]]:
        """Rank-select top N positions by probability from parsed localization dict.

        Returns (positions, probabilities) sorted by descending probability,
        keeping only the top N — matching the OG R script's rank-then-select logic.
        """
        if not loc_dict or n <= 0:
            return [], []
        # Sort by probability descending, break ties by position (lower first)
        ranked = sorted(loc_dict.items(), key=lambda x: (-x[1], x[0]))
        top_n = ranked[:n]
        positions = [pos for pos, _ in top_n]
        probabilities = [prob for _, prob in top_n]
        return positions, probabilities

    def _parse_localization_probabilities(self, loc_string: str) -> Dict[int, float]:
        """Parse EG.PTMLocalizationProbabilities into {peptide_position: probability}.

        Example input:
            '_PVS[Phospho (STY): 92.3%]PS[Phospho (STY): 7.6%]S[Phospho (STY): 0.1%]..._'
        Returns:
            {3: 0.923, 5: 0.076, 6: 0.001, ...}
        """
        if pd.isna(loc_string) or not isinstance(loc_string, str):
            return {}

        s = loc_string.strip("_.*  ")
        result = {}
        pos = 0   # current amino-acid position (1-indexed)
        i = 0     # string cursor

        while i < len(s):
            if s[i] == '[':
                end = s.index(']', i)
                bracket_content = s[i + 1:end]
                if bracket_content.startswith('Phospho (STY)'):
                    match = re.search(r':\s*([\d.]+)%', bracket_content)
                    if match:
                        result[pos] = float(match.group(1)) / 100.0
                i = end + 1
            elif s[i].isalpha():
                pos += 1
                i += 1
            else:
                i += 1

        return result

    def _finalize_collapsed_output(self, data: pd.DataFrame, collapse_level: str) -> pd.DataFrame:

        if collapse_level == "P":
            result = self._process_protein_level_output(data)
        elif collapse_level == "PG":
            result = self._process_protein_group_output(data)
        else:
            raise ValueError(f"collapse_level must be 'P' or 'PG', got '{collapse_level}'")
        return result

    def _process_protein_level_output(self, data: pd.DataFrame) -> pd.DataFrame:

        data["Protein_name"] = data["PTM_Collapse_key"].apply(lambda row: row.split("~")[0])
        data["Protein_group"] = data["Protein_name"].str.split(";")
        data["Protein_group"] = data["Protein_group"].apply(lambda row: row[0])
        data["Protein_name"] = data["Protein_name"].str.split(";")

        data["PTM"] = data["PTM_Collapse_key"].apply(lambda row: row.split("~")[1])
        data[["Gene_name", "Site", "Mult"]] = data["PTM"].str.split("_", expand=True)
        data["Gene_name"] = data["Gene_name"].str.split(";")
        data["Gene_group"] = data["Gene_name"].apply(lambda row: row[0])

        data = data.drop(["Gene_name", "PTM", "PTM_Collapse_key", "PG.Genes", "PG.ProteinGroups"], axis=1)
        data = data.explode("Protein_name", ignore_index=True)

        data["PTM_Collapse_key"] = (
            data["Protein_group"] + "~" + data["Gene_group"] + "_" + data["Site"] + "_" + data["Mult"]
        )
        data["Protein_Collapse_key"] = (
            data["Protein_name"] + "~" + data["Gene_group"] + "_" + data["Site"] + "_" + data["Mult"]
        )

        data = data.drop(["Site", "Mult"], axis=1)
        data["PTM_Collapse_key"] = data["PTM_Collapse_key"].str.replace("#", "_", regex=False)
        data["Gene_group"] = data["Gene_group"].str.replace("#", "_", regex=False)
        return data

    def _process_protein_group_output(self, data: pd.DataFrame) -> pd.DataFrame:

        data["Protein_name"] = data["PTM_Collapse_key"].apply(lambda row: row.split("~")[0])
        data["Protein_group"] = data["Protein_name"].str.split(";")
        data["Protein_group"] = data["Protein_group"].apply(lambda row: row[0])

        data["PTM"] = data["PTM_Collapse_key"].apply(lambda row: row.split("~")[1])
        data[["Gene_name", "Site", "Mult"]] = data["PTM"].str.split("_", expand=True)
        data["Gene_name"] = data["Gene_name"].str.split(";")
        data["Gene_group"] = data["Gene_name"].apply(lambda row: row[0])

        data = data.drop(
            ["Gene_name", "Protein_name", "PTM", "PTM_Collapse_key", "PG.Genes", "PG.ProteinGroups"],
            axis=1
        )

        data["PTM_Collapse_key"] = (
            data["Protein_group"] + "~" + data["Gene_group"] + "_" + data["Site"] + "_" + data["Mult"]
        )

        data = data.drop(["Site", "Mult"], axis=1)
        data["PTM_Collapse_key"] = data["PTM_Collapse_key"].str.replace("#", "_", regex=False)
        data["Gene_group"] = data["Gene_group"].str.replace("#", "_", regex=False)
        return data

    def _load_fasta_to_dict(self, fasta_path: str) -> dict:

        try:
            with open(fasta_path, "r") as file:
                fasta_dict = {}
                current_id = None
                current_sequence = []

                for line in file:
                    line = line.strip()
                    if line.startswith(">"):
                        if current_id is not None:
                            fasta_dict[current_id] = "".join(current_sequence)

                        try:
                            header_parts = line[1:].split("|")
                            if len(header_parts) >= 2:
                                current_id = header_parts[1]
                            else:
                                current_id = line[1:].split()[0]
                            current_sequence = []
                        except Exception as e:
                            raise ValueError(f"Invalid FASTA header format: {line}") from e

                    elif line and current_id is not None:
                        current_sequence.append(line.upper())

                if current_id is not None:
                    fasta_dict[current_id] = "".join(current_sequence)

                if not fasta_dict:
                    raise ValueError("No valid protein sequences found in FASTA file")

                return fasta_dict

        except FileNotFoundError:
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        except Exception as e:
            raise ValueError(f"Error reading FASTA file: {e}")

    def _generate_kinase_sequences(self, data: pd.DataFrame, window_size: int = 6) -> pd.DataFrame:
        result_data = data.copy()
        kinase_sequences = []

        success_count = 0
        error_count = 0
        mismatch_count = 0

        for index, row in data.iterrows():
            try:
                protein_id = row["Protein_group"]
                collapse_key = row["PTM_Collapse_key"]
                ptm_part = collapse_key.split("~")[1]
                site_info = ptm_part.split("_")[1]

                amino_acid = site_info[0]
                position = int(site_info[1:])

                kinase_seq = self._create_kinase_sequence(
                    protein_id, position, amino_acid, window_size
                )

                if kinase_seq.startswith(("FASTA_ERROR:", "POSITION_ERROR:")):
                    error_count += 1
                elif kinase_seq.startswith("SEQUENCE_MISMATCH:"):
                    mismatch_count += 1
                    error_count += 1
                else:
                    success_count += 1

                kinase_sequences.append(kinase_seq)

            except Exception as e:
                error_msg = f"PARSING_ERROR: Row {index} - {str(e)}"
                kinase_sequences.append(error_msg)
                error_count += 1

        result_data["kinase_sequence"] = kinase_sequences

        self.logger.info(
            "Kinase sequences: %d success, %d errors (%d mismatches)",
            success_count, error_count, mismatch_count,
        )

        return result_data

    def _create_kinase_sequence(self, protein_id: str, position: int, amino_acid: str, window_size: int = 6) -> str:
        if protein_id not in self.fasta_dict:
            warning_msg = f"FASTA_ERROR: Protein '{protein_id}' not found in FASTA dictionary"
            self.logger.warning(warning_msg)
            return warning_msg

        protein_sequence = self.fasta_dict[protein_id]
        sequence_length = len(protein_sequence)
        zero_indexed_position = position - 1

        if zero_indexed_position < 0 or zero_indexed_position >= sequence_length:
            warning_msg = f"POSITION_ERROR: Position {position} out of bounds for protein '{protein_id}' (length: {sequence_length})"
            self.logger.warning(warning_msg)
            return warning_msg

        actual_amino_acid = protein_sequence[zero_indexed_position]
        if actual_amino_acid != amino_acid.upper():
            warning_msg = f"SEQUENCE_MISMATCH: Expected '{amino_acid}' at position {position} in '{protein_id}', found '{actual_amino_acid}'"
            self.logger.warning(warning_msg)
            return warning_msg

        start_pos = zero_indexed_position - window_size
        end_pos = zero_indexed_position + window_size

        sequence_parts = []

        if start_pos < 0:
            sequence_parts.append("_" * abs(start_pos))
            actual_start = 0
        else:
            actual_start = start_pos

        actual_end = min(end_pos, sequence_length - 1)
        sequence_parts.append(protein_sequence[actual_start:zero_indexed_position])
        sequence_parts.append(f"*{amino_acid.upper()}*")
        sequence_parts.append(protein_sequence[zero_indexed_position + 1 : actual_end + 1])

        if end_pos >= sequence_length:
            missing_chars = end_pos - sequence_length + 1
            sequence_parts.append("_" * missing_chars)

        kinase_sequence = "_" + "".join(sequence_parts) + "_"
        return kinase_sequence

    def _clean_and_reformat_phospho_data(self, df: pd.DataFrame) -> pd.DataFrame:
        metadata_cols = [
            "UPD_seq", "PTM_localization", "Protein_group", "Gene_group",
            "PTM_Collapse_key", "kinase_sequence"
        ]
        sample_cols = [col for col in df.columns if col not in metadata_cols]

        df_melted = df.melt(
            id_vars=["PTM_Collapse_key"],
            value_vars=sample_cols,
            var_name="sample_name",
            value_name="intensity",
        )

        df_reformatted = df_melted.pivot_table(
            index="sample_name", columns="PTM_Collapse_key", values="intensity", aggfunc="first"
        )

        df_reformatted.columns.name = None
        df_reformatted.index.name = None
        return df_reformatted
