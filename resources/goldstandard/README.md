# Kinase-activity inference gold standard

This directory bundles the field-standard kinase-activity inference
benchmark from Hernández-Armenta et al. 2017 and the Ochoa et al. 2016
Atlas of Human Kinase Regulation. Used by
`alphaphos.enrichment.validation` and `scripts/validate_enrichment_ev3.py`.

All files are open-access (CC-BY 4.0) via published Supplementary tables
and the EBI BioStudies archive.

## Files

| File | Content | Source |
| --- | --- | --- |
| `goldstandard_HernandezArmenta2017_kinase_conditions.csv` | 27 condition groups → 30 kinases + direction | Table 1 of Hernández-Armenta et al. 2017, *Bioinformatics* 33(12):1845, `10.1093/bioinformatics/btx082` |
| `goldstandard_OchoaAtlas2016_EV3_expected_regulation.csv` | 132 (condition → kinase → direction) pairs with PubMed IDs | Ochoa et al. 2016 *Mol Syst Biol* Atlas Table EV3, BioStudies **S-EPMC5199121** |
| `reference_OchoaAtlas2016_EV2_KSEA_activities.csv` | Reference kinase-activity matrix (215 kinases × 399 conditions) computed by the atlas's own KSEA | ibid Table EV2 |
| `OchoaAtlas2016_EV1_conditions_metadata.csv` | Metadata for 435 conditions (PMID, cell line, treatment, MS setup) | ibid Table EV1 |

## Citation

Please cite the original papers when using this benchmark:

> Hernández-Armenta C, Ochoa D, Gonçalves E, Saez-Rodriguez J,
> Beltrao P. **Benchmarking substrate-based kinase activity inference
> using phosphoproteomic data.** *Bioinformatics* 33(12):1845-1851
> (2017). `doi:10.1093/bioinformatics/btx082`

> Ochoa D, Jonikas M, Lawrence RT, El Debs B, Selkrig J, Typas A,
> Villén J, Santos SDM, Beltrao P. **An atlas of human kinase
> regulation.** *Mol Syst Biol* 12(12):888 (2016).
> `doi:10.15252/msb.20167295`
