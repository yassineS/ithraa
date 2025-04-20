## TODO list for the project

### FEATURES
- [ ] Github pages.
- [ ] Move plotting to main pipeline (subcommand).
- [ ] Setup github pipelines:
	- [ ] Pytest.
	- [ ] Coverage badge.
	- [ ] Build documentation/website.
- [x] Rotate scores within Chromosome.
- [ ] Mahalanobis distance matching.
	- [ ] Output PCA/tSNE of set (VIPs) genes and matching genes across iterations.
- [ ] For very large datasets or extremely long runs, we could consider implementing a checkpoint/resume feature in the future if needed.
- [ ] Plotting:
	- [ ] Fixing plots.
	- [ ] Moving plotting function to main pipeline.
	- [ ] Making plotting optional.
- [ ] Testing vs. perl.
- Fix output:
	- [ ] Output summary table.
	- [ ] Output full table instead of json.
	- [ ] Output config in toml instead of json.
	- [ ] Don't output plot folder unless requested.

### BUG FIXES and Core Functionality
- [ ] Check if factors used properly in matching.
- [ ] Documentation.
- [x] Example data.
- [x] Check and clean parameters that don't do anything anymore:
	- [x] fdr.interrupted.
	- [x] fdr.shuffling_segments.
	- [x] bootstrap.runs.
	- [x] bootstrap.simultaneous_runs.