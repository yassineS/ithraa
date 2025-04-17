### TODO list for the project

### FEATURES
- [ ] Github pages.
- [ ] Rotate scores within Chromosome.
- [ ] Mahalanobis distance matching.
- [ ] For very large datasets or extremely long runs, we could consider implementing a checkpoint/resume feature in the future if needed.

### BUG FIXES and Core Functionality
- [ ] Documentation.
- [ ] Example data.
- [ ] Testing vs. perl.
- [] Check and clean parameters that don't do anything anymore:
	- [x] fdr.interrupted.
	- [x] fdr.shuffling_segments.
	- [x] bootstrap.runs.
	- [ ] bootstrap.simultaneous_runs.
- [x] Fix memory management issues in parallel processing.
- [x] Improve error handling when thresholds fail to process.
- [x] Switch to faster libries:
	- [x] polars.
	- [x] numba.
- [x] Parallelise FDR.
- [x] CLI.
- [x] Pytest.
- [x] Parallelise the "Processing thresholds" steps.
- [x] Fix progress bars.
	- [x] Don't print thresholds to console, but write to log file.
- [x] Add timestamp to logfile name.
- [x] what does "interest" parameter in input do?
- [x] Use "population" parameter to select a subset of populations from the input ranks file.
- [x] Pipeline execution failed: 'GeneSetEnrichmentPipeline' object has no attribute 'save_results'.
- [x] Check and clean if necessary the legacy parameters:
	- [x] prefix = "all_ihsfreqafr_ranks"
	- [x] sizes = ["50kb", "100kb", "200kb", "500kb", "1000kb"]
