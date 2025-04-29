l# Transitioning from CUDA to SYCL

Join us on April 29, 2025, for a webinar covering the process of porting CUDA code to SYCL, with a focus on high-performance math libraries like cuBLAS and cuFFT. ALCF's Thomas Applencourt and Abhishek Bagusetty will discuss key challenges, such as differences in API, memory management, and execution models, and provide strategies for achieving portability and performance. 

Special attention will be given to common pitfalls in porting, including synchronization issues and device memory handling. Additionally, we'll explore how to optimize SYCL code for Aurora's advanced architecture, highlighting techniques to avoid bottlenecks, map kernels efficiently, and leverage multi-level parallelism. Through case studies and practical examples, this talk will guide developers transitioning CUDA applications to SYCL without sacrificing performance, particularly on heterogeneous platforms like Aurora.

Welcome to the ALCF Aurora setup guide! This document provides step-by-step instructions for accessing Aurora, setting up your development environment, and cloning the necessary project repositories.

---

## 1. Accessing ALCF Aurora

To access Aurora, you must have:
- An approved project allocation at [ALCF](https://www.alcf.anl.gov/)
- A valid ALCF user account
- Duo two-factor authentication set up

**SSH Access:**
```bash
ssh <your_username>@aurora.alcf.anl.gov
```
If this is your first login, you may need to complete initial key setup and security procedures. For more details, see the [ALCF Accounts and Access Guide](https://www.alcf.anl.gov/support-center/finding-your-way/accounts-access).

---

## 2. Setting Up Your Environment

After logging in, you must load the appropriate modules to configure the programming environment for Aurora. By default, all necessary modules for Intel's oneAPI are available. Additionally, users must set up the CUDA headers to enable proper operation of porting tools from CUDA to SYCL.

### Load Environment Modules
```bash
module use /soft/modulefiles
module load cmake
module load headers/cuda/12.0.0
```

## 3. Cloning Required Projects

Clone the following repositories needed for your work:

```bash
git clone https://github.com/your-org/your-main-project.git
git clone https://github.com/your-org/your-helper-library.git
```

Replace the URLs above with the actual project repository links you have been provided. If you require access, ensure your GitHub account is added to the relevant organization or team.

---

## 4. Build Instructions

Typical build workflow:

```bash
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx
make -j
```

Make sure to adjust compiler options if you need to use MPI or SYCL backends.

---

## 5. Need Help? Contact Us

For any technical questions or help setting up:

- **Abhishek Bagusetty**  
  - Email: abagusetty@anl.gov  
  - GitHub: [abagusetty](https://github.com/abagusetty)

- **Thomas Applencourt**  
  - Email: tapplenc@anl.gov  
  - GitHub: [tapplencourt](https://github.com/tapplencourt)

Feel free to reach out if you encounter issues with access, setup, builds, or performance tuning!

---

## Additional Resources

- [ALCF Aurora User Guide](https://docs.alcf.anl.gov/aurora/)
- [Intel oneAPI Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
- [Cray Programming Environment](https://docs.hpe.com/)

---



#Transitioning from CUDA to SYCL
Source on Aurora hardware: https://docs.alcf.anl.gov/aurora/#aurora-machine-overview

## Contributing to documentation

### Setting up Environment

Setup necessary modules such that 
```bash
module use /soft/modulefiles
module load cmake
module load headers/cuda/12.0.0
```

### Samples to Clone

Clone the following projects 
To build documentation locally, you need a Python environment with `mkdocs` installed.  Check that Python 3.6+ is installed:

Using Git's SSH protocol. Make sure you add your SSH public key to your GitHub account:
```bash
git clone git@github.com:argonne-lcf/user-guides.git
cd user-guides
git submodule init; git submodule update
```

### Installing MkDocs

To install `mkdocs` in the current environment: 
```bash
cd user-guides
make install-dev
```

### Preview the docs locally and test for errors

Run `mkdocs serve` or `make serve` to auto-build and serve the docs for preview in your web browser:
```bash
make serve
```

GitHub Actions are used to automatically validate all changes in pull requests before they are merged, by executing `mkdocs build --strict`. The [`--strict`](https://www.mkdocs.org/user-guide/configuration/#validation) flag will print out warnings and return a nonzero code if any of a number of checks fail (e.g. broken relative links, orphaned Markdown pages that are missing from the navigation sidebar, etc.). To see if your changes will pass these tests, run the following command locally:
```
make build-docs
```

### Working on documentation

* All commits must have a commit message
* Create your own branch from the `main` branch.  Here, we are using `YOURBRANCH` as an example:
```bash
cd user-guides
git fetch --all
git checkout main
git pull origin main
git checkout -b YOURBRANCH
git push -u origin YOURBRANCH
```
* Commit your changes to the remote repo:
```bash
cd user-guides
git status                         # check the status of the files you have edited
git commit -a -m "Updated docs"    # preferably one issue per commit
git status                         # should say working tree clean
git push origin YOURBRANCH         # push YOURBRANCH to origin
git checkout main                  # move to the local main
git pull origin main               # pull the remote main to your local machine
git checkout YOURBRANCH            # move back to your local branch
git merge main                     # merge the local develop into **YOURBRANCH** and
                                     # make sure NO merge conflicts exist
git push origin YOURBRANCH         # push the changes from local branch up to your remote branch
```
* Create merge request from https://github.com/argonne-lcf/user-guides from `YOURBRANCH` to `main` branch.

## Inbound Links Validation
External URLs pointing to our docs are tracked in [includes/validate-inbound-URLs.txt](includes/validate-inbound-URLs.txt) and validated during build to prevent broken links from the main ALCF site, etc. Add URLs to that file to ensure that the matching `.md` in this repository is never moved, renamed, or deleted.

There are two Python `scripts/` that perform this function:
1. (works with `mkdocs build` and `serve`): Translate URLs to relative path links to their matching source `.md` files; write these links to `docs/inbound-links.md`. Use MkDocs' built-in validation (adding `--strict` flag when running `mkdocs build`, in order to return an error code if they are invalid).
2. (works with `mkdocs build`, only): Use a lightweight post-build validation on generated `site/` directory HTML contents.

## Contact Us
Abhishek Bagusetty, abausetty@anl.gov
Thomas Applencourt, tapplencourt@anl.gov
