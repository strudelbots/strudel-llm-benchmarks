

# strudel-llm-benchmarks

This repository provides a set of benchmarks designed to evaluate large language models' (LLMs) 
ability to understand code.

## Introduction

1. This repository is a work in progress (WIP) and subject to continuous updates.
2. The `root` directory contains executable file(s). Before running them, ensure that you have set the necessary environment variables (ENV vars). If any required variables are missing, the scripts will provide appropriate error messages.
3. The `code` directory contains the source code for the project.

## How It Works

### `run_file_summarizer`

The `run_file_summarizer.py` script aims to generate a concise summary of a 
Python file’s core functionality.

#### Usage
- Set the `REPO_DIR` environment variable to the base directory of your repository.
- The main function will read Python files from this directory and use your selected 
LLM model to generate a summary for each file.
- Each summary is stored in a separate JSON file within `JSON_FILES_DIRECTORY`. 
The default directory is `/tmp`.

#### Output
The main function outputs JSON files containing summaries of the analyzed Python files.
