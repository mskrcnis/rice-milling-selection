---
output:
  pdf_document: default
  html_document: default
editor_options: 
  markdown: 
    wrap: 72
---

## Provenance for this README

-   File name: README.md
-   Authors: Rebollo I, Scheffel S, Blanco P, Molina F, Martínez S,
    Carracelas G, Pérez de Vida F, Rosas JE.
-   Date created: 2023-02-14
-   Date modified: 2023-02-16

## Dataset Version and Release History

-   Current Version:
    -   Number: 1.0.0
    -   Date: 2023-02-16
    -   Persistent identifier: DOI: <doi:10.5061/dryad.x69p8czn8>
    -   Summary of changes: n/a
-   Embargo Provenance: n/a
    -   Scope of embargo: n/a
    -   Embargo period: n/a

## Dataset Attribution and Usage

-   Dataset Title: Instituto Nacional de Investigación Agropecuaria
    (INIA) Rice Breeding Program Historical Dataset

-   Persistent Identifier:
    [https://](https://doi.org/10.5061/dryad.x69p8czn8){.uri}[doi.org/10.5061/dryad.x69p8czn8](https://doi.org/10.5061/dryad.x69p8czn8)

-   Dataset Contributors:

    -   Creators: Inés Rebollo, Sheila Scheffel, Pedro Blanco, Federico
        Molina, Sebastián Martínez, Gonzalo Carracelas, Fernando Pérez
        de Vida, Juan E. Rosas.

-   Date of Issue: 2023-02-16

-   Publisher: Instituto Nacional de Investigación Agropecuaria (INIA
    Uruguay)

-   License: Use of these data is covered by the following license:

    -   Title: CC0 1.0 Universal (CC0 1.0)
    -   Specification:
        <https://creativecommons.org/publicdomain/zero/1.0/>; the
        authors respectfully request to be contacted by researchers
        interested in the re-use of these data so that the possibility
        of collaboration can be discussed.

-   Suggested Citations:

    -   Dataset citation: Rebollo, Inés, Scheffel, Sheila, Blanco,
        Pedro, Molina, Federico, Martínez, Sebastián, Carracelas,
        Gonzalo, Pérez de Vida, Fernando, Rosas, Juan E. (2023),
        Instituto Nacional de Investigación Agropecuaria (INIA) Rice
        Breeding Program Historical Dataset, Dryad, Dataset,
        <https://doi.org/10.5061/dryad.2bvq83btg>

    -   Corresponding publication: Manuscript under review in Crop
        Science with the ID CROP-2022-10-0677-ORA.R1

## Contact Information

-   Name: Juan E. Rosas

-   Affiliations: Area or Plant Breeding and Biotechnology, Instituto
    Nacional de Investigación Agropecuaria (Uruguay)

-   ORCID ID: <https://orcid.org/0000-0001-6129-8304>

-   Email: [jrosas\@inia.org.uy](mailto:jrosas@inia.org.uy){.email}

-   Contributor ORCID IDs:

    -   Inés Rebollo: <https://orcid.org/0000-0002-9899-0801>
    -   Sheila Scheffel: <https://orcid.org/0000-0001-7637-0321>
    -   Gonzalo Carracelas: <https://orcid.org/0000-0002-4306-1546>
    -   Sebastián Martínez : <https://orcid.org/0000-0003-0455-5823>

-   <div>

    ------------------------------------------------------------------------

    </div>

# Additional Dataset Metadata

## Acknowledgements

-   Funding sources: INIA (Projects AZ35, AZ13, and fellowships to IR
    and SS; Agencia Nacional de Investigación Agropecuaria (grant
    ANII-FSDA_1\_2018_1\_154120); Comité Académico de Posgrado
    (fellowships to IR and SS). We acknowledge the effort of IRBP's
    field and laboratory technical personnel.

------------------------------------------------------------------------

# Methodological Information

-   A detailed description on data collection and processing is
    available in the corresponding manuscript submitted to Crop Science
    journal. Briefly, for data unification a uniform format spreadsheet
    was designed with standardized names for fields, creating one
    spreadsheet per trial. An R code was created and used to merge
    trial's data into a single table. All tables were then unified into
    one with all the available IRBP information. A relational database
    structure was defined by identifying the elements that constituted
    the system and the key fields that connected them. Variable names
    and levels of all categorical variables were standardized.

-   <div>

    ------------------------------------------------------------------------

    </div>

# Data and File Overview

## Summary Metrics

-   File count: 5
-   Total file size: 67.5 MB
-   Range of individual file sizes: 62.8 KB - 56.3 MB
-   File formats: .txt

## Table of Contents

-   Trials
-   Phenotypes
-   Lines
-   GenomicInformation
-   SNP

## INIA´s Rice Breeding Program Dataset:

Historical dataset of the rice breeding program of the National
Agricultural Research Institute of Uruguay, from 1997 to 2020.

**Phenotypic and pedigree data**

Original data consisted of records from 1997 to 2020 of field trials,
blast nurseries, and laboratory analyses of milling and cooking quality
for fixed inbred lines from the IRBP Indica, Tropical japonica,
Temperate japonica and Clearfield germplasm types. Records from field
trials correspond to the E1, E2, E3, E4 and FE evaluation stages. Data
included trial information (location, sowing and harvest date), records
of phenotypic variables (grain yield (GY), plant height (PH), flowering
date (FD), leaf blast (LB, caused by *Pyricularia oryzae*), *Nakataea
oryzae* resistance (NO), *Rhizoctonia oryzae-sativae* resistance (ROS),
yield after milling (YAM), proportion of head rice recovery (PHR), grain
chalkiness (GC), stained milled grain (SM), amylose content (AC), alkali
digestion value (ADV), line and pedigree information, and genomic
information for selected lines. Raw data is stored in spreadsheets, with
Indica and Temperate japonica in a wide range of formats and Tropical
japonica and Clearfield in two distinct types of spreadsheets: type A
with experimental line ID and pedigree records, and type B with
phenotypic records for the variables measured in each trial.

**Genomic information**

Two batches of genomic information obtained by genotyping-by-sequencing
with the ApeKI enzyme (Elshire et al., 2011) were made compatible,
resulting in a set of 61,260 single nucleotide polymorphisms (SNPs) for
965 selected lines. The first batch consisted of sequences of 601 lines
that represented the variability of IRBP's elite germplasm in 2011 and
were obtained at Cornell University's Biotechnology Resource Center with
a 100bp single-end sequencer (Rosas et al., 2018). The second batch
comprised sequences of 364 lines for which there were seeds available in
INIA's gene bank, selected according to two criteria: 1) lines that were
evaluated in yield trials representing contrasting environments based on
their climatic conditions, and 2) lines that were most widely used as
parents. The second batch was sequenced in 2020 with an Illumina 150bp
paired-end sequencer at the University of Wisconsin's DNA Sequencing
Facility. Raw sequences of the two batches were processed together with
the TASSEL 5.0 pipeline (Bradbury et al., 2007) and were aligned to the
Nipponbare reference genome 7.0 (Kawahara et al., 2013) with the BWA
alignment package (Li & Durbin, 2009). SNPs with minor alternate allele
frequency ≥ 0.03, ≤ 50% of missing data and ≤15% of observed
heterozygosis were retained. Remaining missing data were imputed with
BEAGLE software (Browning et al., 2018) and the final matrix was
transformed to numeric coding (0, 1, 2) for analyses.

## Description of the Data and file structure

The dataset is structured in the following tables: Trials, Phenotypes,
Lines, Genomic_Information, and SNP, all in txt format. Missing data is
coded as NA. Detailed information can be found in the corresponding
paper submitted to Crop Science (Manuscript code
CROP-2022-10-0677-ORA.R1). Briefly, the Trials table has records of 996
trials from 12 locations over a span of 23 years; the Phenotypes table
has records of 91,636 field plots for 14 phenotypic variables; the Lines
table has 20,628 records of breeding lines, cultivars used as checks,
and their parents; the GenomicInformation table has records of 965
lines; and the SNP table has records of 61,260 SNP markers.

## Trials table

This table has the following information:

YEAR: Trial's harvest year (e.g.: 2020 is season 2019-2020)\
TRIAL: String identifying the trial within each year and location\
LOCATION: Location where the trial was conducted\
SOWING_DATE: Trial's sowing date. Format: yyyy-mm-dd\
HARVEST_DATE: Trial's harvest date. Format: yyyy-mm-dd\
EVALUATION_STAGE: Trial's evaluation stage (E1 to E4, first to fourth
years of field evaluation; FE, followed final field evaluation)\
HERITABILITY: Trial's broad sense heritability of grain yield\
CV: Trial's coefficient of variation for grain yield\
GENETIC_VARIANCE: Trial's genetic variance for grain yield\
RESIDUAL_VARIANCE: Trial's residual variance for grain yield

## Phenotypes table

This table has the following information:

YEAR: Trial's harvest year (e.g.: 2020 is season 2019-2020)\
TRIAL: String identifying the trial within each year and location\
LOCATION: Location where the trial was conducted\
BLOCK: Block number of the plot in the trial\
LINE_ID: Unique string identifying the line\
GY (Grain yield): Grain yield (kg ha^-1^)\
PH (Plant height): Plant height, from soil surface to the tip of the
flag leaf (cm)\
FD (Flowering date): Date when 50% of plants are at heading stage\
DTF (Days to flowering): Number of days between sowing and flowering
dates (days)\
LB (Leaf blast): Leaf blast caused by Pyricularia oryzae rating at the
nursery using SES IRRI scale\*\
NO (Nakataea oryzae): Stem rot rating in yield plots using modified SES
IRRI scale\*\
ROS (Rhizoctonia oryzae-sativae): Aggregate sheath spot rating in yield
plots using modified SES IRRI scale\*\
YAM (Yield after milling): Percentage of milled rice kernels (%)\
PHR (Proportion of head rice recovery): Percentage of milled whole rice
kernels (%)\
GC (Grain chalkiness): Percentage of chalky endosperm milled kernels
(%)\
SM (Stained milled grain): Percentage of stained milled kernels (%)\
LW_RATIO: Ratio between grain length and width\
AC (Amylose content): Percentage of apparent amylose content (%)\
ADV (Alkali digestion value): Milled kernel dispersion in potassium
hydroxide using SES IRRI scale\*

\*SES IRRI Scale: Standard Evaluation System for Rice of the
International Rice Research Institute's (IRRI, 2002)

## Lines table

This table has the following information:

LINE_ID: Unique string identifying the line\
P1: String identifying maternal line\
P2: String identifying paternal line\
GENERATION: Number of generations advanced through self-pollination\
GERMPLASM_TYPE: String identifying the germplasm and grain quality type
(Indica, long-grain indica; Tropical japonica, long-grain tropical
japonica; Temperate japonica, short, middle, and heavy grain temperate
japonica; Clearfield, herbicide resistant Indica and Tropical japonica
varieties for the Clearfield production system)\
BANK_ID: Unique string identifying the line in INIA's gene bank\
FLAR_ID: Unique string identifying the line from the Latin American Fund
for Irrigated Rice (FLAR)

## Genomic information table

This table has the following information:

LINE_ID: Unique string identifying the line\
GENOMIC_INFORMATION: String with numerically coded genomic information
(0, 1, 2) for SNPs

## SNPs table

This table has the following information:

SNP_ID: String identifying the SNP marker with its chromosome and
position CHROMOSOME: Numeric value for the chromosome (1 to 12)\
POSITION: Numeric value for the SNP position in base pairs

## References

IRRI. (2002). Standard Evaluation System For Rice. International Rice
Research Institute.
