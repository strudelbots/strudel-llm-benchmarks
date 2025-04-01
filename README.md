

# strudel-llm-benchmarks

This repository provides a set of benchmarks designed to evaluate large language models' (LLMs) 
ability to understand code.
### How it Works 
The main function (in `run_file_smmmarizer.py`) selects files from a repository and 
submits them to your preferred LLM(s) for analysis. 
The model then examines each file and summarizes its **core functionality** in up to three sentences.
#### Here is an example output: 
File: `/torch/distributed/checkpoint/logging_handlers.py` from the Pytorch repository: 
1. gpt-4o: *"The provided Python file initializes and configures a logging system by integrating a custom logger named \"dcp_logger\" into PyTorch's distributed logging handlers. It assigns a `NullHandler` to \"dcp_logger\" to ensure log messages with this logger are ignored unless explicitly handled elsewhere. The file appears to set up a placeholder logger for distributed computing purposes without introducing any additional functionality."*,
2. gpt-35-turbo: *"The provided Python file defines a module for handling logging in a distributed environment with the torch library. It initializes a logging handler for a specific logger named \"dcp_logger\" using a NullHandler to discard all log messages. The file also updates the existing log handlers with the newly defined handler."*
3. gpt-4: *"The Python file named \"logging_handlers.py\" primarily configures logging handlers for a specific logger named \"dcp_logger\". It imports the logging module and a specific function or module named _log_handlers from the torch.distributed.logging_handlers package. The main functionality is updating the _log_handlers dictionary to set a 'NullHandler' for the 'dcp_logger', effectively ensuring that any logs sent to this logger will be ignored or not outputted anywhere by default.".*

This repository is a work in progress (WIP) and subject to continuous updates.
## How It Works

### `run_file_summarizer`

The `run_file_summarizer.py` script aims to generate a concise summary of a 
Python file’s core functionality.

#### How to Run it.  
- See `__init__.py` in  `benchmark_code` directory and set the environment vars.
- Set up initial parameters in `__main__ of` in `run_file_summarizer` 
(e.g., the models you want to use) ans sample rte   

#### Output
The main function generates JSON files, each containing a summary of a single file for each requested model.
At the end of the process, the script produces a final 
JSON file that consolidates all summaries from all models (e.g., see `04-01-2025__gpt-35-turbo_gpt-4_gpt-4o__pytorch2_summary.json`).

