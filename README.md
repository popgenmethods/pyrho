# pyrho

Fast demography-aware inference of fine-scale recombination rates based on
fused-LASSO


## Table of Contents
* [Human Recombination Maps](#human-recombination-maps)
* [Human Population Sizes](#human-population-sizes)
* [Installation](#installation)
* [Usage](#usage)
    * [make_table](#make_table)
    * [hyperparam](#hyperparam)
    * [optimize](#optimize)
    * [compute_r2](#compute_r2)
* [Example](#example)
* [Citation](#citation)


NB: in version 0.1.0 we switched everything from coalescent units to
natural scale -- that is inputs should now be in terms of Ne, generations,
and per-base, per-generation mutation rates.  The output is now also
automatically scaled to be the per-generation recombination rate.

## Human Recombination Maps

We have inferred recombination maps for each of the 26 populations in phase 3
of the 1000 Genomes Project.  Those maps are available
[at this link](https://drive.google.com/open?id=1Tgt_7GsDO0-o02vcYSfwqHFd3JNF6R06)
including maps inferred for the two most recent genome builds
hg19/GRCh37 and hg38/GRCh38. 

The recombination maps for hg38/GRCh38 are now also available for simulations using
the wonderful [stdpopsim](https://github.com/popsim-consortium/stdpopsim) package.

## Human Population Sizes

When making the recombination maps for the 1000 Genomes Project populations we used
[smc++](https://github.com/popgenmethods/smcpp)
to infer population size histories for each population.
Those size histories are plotted in
[Figure 2 of the pyrho paper](https://advances.sciencemag.org/content/5/10/eaaw9206/),
and the data used to make those plots are available in `smcpp_popsizes_1kg.csv`.
The `x` column is time in years
assuming a generation time of 29 years,
and the `y` column
contains the population size in Ne.


Installation
------------

pyrho is compatible with both python 2 and python 3.
pyrho makes use of a number of external packages including the excellent
[numba](https://github.com/numba/numba),
[msprime](https://github.com/tskit-dev/msprime),
and
[cyvcf2](https://github.com/brentp/cyvcf2)
packages.  As such it is recommended to install pyrho in a
virtual environment.

If using
[conda](https://conda.io/en/master/)
this can be accomplished by running

```
conda create -n my-pyrho-env
conda activate my-pyrho-env
```

or using 

```
virtualenv my-pyrho-env
source my-pyrho-env/bin/activate
```

Once you have set up and activated your virtual environment, you must
first install
[ldpop](https://github.com/popgenmethods/ldpop).
Change to a directory where you want to store ldpop and then run

```
git clone https://github.com/popgenmethods/ldpop.git ldpop
pip install ldpop/
```

Note that this will create a directory ldpop.

pyrho makes use of
[msprime](https://github.com/tskit-dev/msprime),
which requires
[gsl](https://www.gnu.org/software/gsl/)
and
[hdf](https://www.hdfgroup.org).
pyrho also has a dependency on
[openssl](https://www.openssl.org).
If you do not have these installed, these can be installed using
```apt-get```, ```yum```, ```conda```, ```brew``` etc...

For example, to install openssl on Ubuntu run:

```
sudo apt-get install libssl-dev
```

You will also need to have cython installed.  If you do not yet have it
installed, run

```
pip install cython
```

You should be able to then just clone and install pyrho by running

```
git clone https://github.com/popgenmethods/pyrho.git pyrho
pip install pyrho/
```

You can check that 
everything is running smoothly by running

```python -m pytest pyrho/tests/tests.py```

NB: the first time you run pyrho, numba will compile and cache a number
of functions, which can take up to ~30 seconds.

Usage
-----

pyrho has a command line interface and consists of a number of separate
commands.  A typical workflow is to first use
[make_table](#make_table)
to build a lookup table and then use
[hyperparam](#hyperparam)
to find reasonable hyperparameter settings for your data.  Finally, use
[optimize](#optimize)
to infer a fine-scale recombination map from data.  There is also a command,
[compute_r2](#compute_r2),
that computes statistics of the theoretical distribution of r<sup>2</sup>.

### make_table

Before performing any inference, you must first compute a lookup table.  A
standard use case would be

```
pyrho make_table --samplesize <n> --approx  --moran_pop_size <N> \
--numthreads <par> --mu <mu> --outfile <outfile> \
--popsizes <size1>,<size2>,<size3> --epochtimes <breakpoint1>,<breakpoint2>
```

which indicates that we should compute a lookup table for a sample of size
```<n>```
, from a population where at present the size is 
```<size1>```
, at 
```<breakpoint1>```
generations in the past the size was 
```<size2>```
and so on, with a per-generation
mutation rate 
```<mu>```
The --numthreads option tells pyrho to use 
```<par>```
processors
when computing the lookup table.  Finally, --approx with --moran_pop_size
tells pyrho to compute an approximate lookup table for a larger sample size
```<N>```
and then downsample to a table for size
```<n>```
.  In general 
```<N>```
should be
about 25-50% larger than
```<n>```
.  Without using the --approx flag, pyrho can
compute lookup tables for 
```<n>```
< ~50, whereas with the --approx flag, pyrho
can handle sample sizes in the hundreds (e.g., 
```<n>```
= 200,
```<N>```
= 256) with little loss in accuracy as long as ```<n>``` << ```<N>```.

The output is an hdf format table containing all of the pre-computed
likelihoods needed to run [hyperparam](#hyperparam), [optimize](#optimize),
and [compute_r2](#compute_r2).

Note that make_table can consume significant amounts of memory (N=256 requires
about 100G of RAM using the --approx flag).

To see a full list of options and their meaning, run 
```pyrho make_table --help```.


### hyperparam


After computing a lookup table with
[make_table](#make_table),
it is a good idea to find reasonable settings for the main hyperparameters
of pyrho: the window size and the smoothness penalty.  This command
simulates data and performs optimization under a number of different
hyperparameter settings and outputs the accuracy in terms of a number of
different metrics.  A typical usage is

```
pyrho hyperparam --samplesize <n> --tablefile <make_table_output> \
--mu <mu> --ploidy <ploidy> \
--popsizes <size1>,<size2>,<size3> --epochtimes <breakpoint1>,<breakpoint2>  \
--outfile <output_file>
```

where ```<n>``` is your haploid sample size, ```<make_table_output>``` is
the output of running [make_table](#make_table), and
```<mu>```, ```<size1>```..., and ```<breakpoint1>```... are as above. 
```<ploidy>``` should be set to ```1``` if using phased data and ```2``` for
unphased genotype data.  Ploidies other than ```1``` or ```2``` are not
currently supported.

The output is a table containing various measures of accuracy of the inferred
maps compared to the recombination maps used in the simulations (drawn from
the [hapmap](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1880871/) recombination map).  The optimal hyperparameters will
depend on your application -- it may be important to maximize a particular
measure of correlation or to minimize the L2 norm between the true and
inferred maps. In any case, you can use the results in the output table to
choose the hyperparameters for running [optimize](#optimize).

To see a full list of options and their meaning, run 
```pyrho hyperparam --help```.


### optimize


After computing a lookup table using [make_table](#make_table) and choosing
reasonable hyperparameters (optionally using [hyperparam](#hyperparam)) you
are ready to infer recombination maps from real data.

pyrho supports data in fasta format and LDhat's sites and locs format, but it
is easiest to directly use VCF formatted data.  pyrho supports VCF, bgzipped
VCF, and BCF format files. If using LDhat's formats see the 
[note about using sites and locs](#a-note-about-ldhat-format-input).

A typical usage is

```
pyrho optimize --vcffile <data> --windowsize <w> --blockpenalty <bpen> \
--tablefile <make_table_output> --ploidy <ploidy> --outfile <output_file> \
--numthreads <par>
```

with ```<data>``` being a VCF, bgzipped VCF, or BCF file containing a
single chromosome, ```<w>``` and
```<bpen>``` being hyperparameters chosen using [hyperparam](#hyperparam),
and ```<ploidy>``` should be ```1``` for phased data and ```2``` for unphased
data.


The output file has three columns -- the first column is the zero-indexed
start of an interval, the second column is the end of the interval
(non-inclusive), and the third column is r, which is the per-base
per-generation recombination rate in that interval.

To see a full list of options and their meaning, run
```pyrho optimize --help```.


#### A note about LDhat format input

The preferred input format for pyrho is a VCF format file containing a single
chrmosome.  LDhat format files may also be used with the following important
caveat.

If using LDhat formatted data (i.e., a sites file and a locs file) note that
we use a slighly different convention than LDhat (sorry!).  For unphased data,
LDhat uses the convention 0 = homozygous reference, 1 = homozygous alternate,
2 = heterozygous.  We do *not* use this convention.

We use 0 = homozygous reference, 1 = heterozygous, 2 = homozygous alternate,
and N = missing.  That is, each entry should be the number of alternate
alleles, or N for missing.
We otherwise follow the formatting (including the headers) of LDhat as
described [here](http://ldhat.sourceforge.net).


### compute_r2

[compute_r2](#compute_r2) computes statistics of the theoretical distribution
of r<sup>2</sup> using a lookup table generated using
[make_table](#make_table).  In particular, it can compute the mean and/or
quantiles of the distribution of r<sup>2</sup>.

A typical usage is

```
pyrho compute_r2 --tablefile <make_table_output> --samplesize <n> \
--quantiles 0.25,0.5,0.75 --compute_mean
```
which will compute the 25th, 50th, and 75th percentiles as well as the mean
of the distribution of r<sup>2</sup> for the lookup table stored in
```<make_table_output>```.  It is possible to compute statistics for a smaller
sample size by setting ```<n>``` to be less than the sample size for which
```<make_table_output>``` was computed.


Example
-------

The example folder contains a well-commented shell script example.sh which
runs through a typical use-case, taking a VCF file and the output of smc++
and ultimately computing a fine-scale recombination map.

Citation
--------

If you use pyrho please cite

[Spence, J.P. and Song, Y.S. Inference and analysis of population-specific fine-scale recombination maps across 26 diverse human populations. Science Advances, Vol. 5, No. 10, eaaw9206 (2019).](https://doi.org/10.1126/sciadv.aaw9206)

and

[Kamm, J.A.\*, Spence, J.P.\*, Chan, J., and Song, Y.S. Two-locus likelihoods under variable population size and fine-scale recombination rate estimation. Genetics, Vol. 203 No. 3 (2016) 1381-1399.](http://www.genetics.org/content/203/3/1381)

