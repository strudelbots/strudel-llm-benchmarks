## Strudel's LLM Benchmarking for Code Comprehension
With all the hype around LLMs promising 10x productivity, I wanted to take a more grounded look at what’s actually being delivered. Benchmarks are everywhere, but like electric vehicle mileage stats, they rarely reflect real-world outcomes.

ROI (Return on Investment) is a straightforward concept: what you gain compared to what you spend. This repository provides a set of benchmarks to evaluate the ROI of large language models.

### What We Do
We randomly selected 100 Python files from the PyTorch repository and tasked approximately 10 different LLMs with summarizing each file in under three sentences. The resulting summaries are publicly available in this repository [Code Comprehension Summaries](https://raw.githubusercontent.com/strudelbots/strudel-llm-benchmarks/refs/heads/main/results/pytorch_DB.json).

Using these summaries (and their associated metadata), we can now begin comparing ROI for code comprehension using practical metrics such as verbosity, latency, cost, perceived accuracy, and information gain. This allows us to assess how these models actually perform—and what that says about their return on investment.


## How to Contribute

1. Don’t buy me a coffee.
2. Instead, use that \$5-worth of energy to contribute new file summaries to the results. See [How It Works](#how-it-works) for details.
3. Once you've analyzed your selected files (with your selected models), run the `from_cache_to_db.py` script to update the main JSON results file.
4. This repo is always a work in progress. If you spot an issue, please fix it!



### How It Works
The main script (`generate_file_summaries.py`) selects files from a repository and submits them to your chosen LLM(s) for analysis.

To run the script, ensure all required environment variables are set (refer to `__init__.py` in the `benchmark_code` directory for details). You may also need to update some hardcoded access points to integrate your models. Additionally, a caching mechanism is implemented to avoid redundant LLM calls. When you run the script again on the same files, no LLM calls are made. 

One can also use to generate embedding for each summary `generate_embeddings_db_from_summaries_b.py.py` 

### Example: *Core Functionality* Summarized by Three Models for a Single PyTorch File

File: `/torch/distributed/checkpoint/logging_handlers.py`:  
1. **Gemini-2.5**: *"This Python script configures logging by defining a specific logger named 
'dcp_logger'. It then updates a central dictionary (`_log_handlers`), likely used within the 
`torch.distributed` library, to associate this logger name with a `logging.NullHandler`. 
This effectively ensures that log messages from 'dcp_logger' are discarded by default 
unless another handler is explicitly configured elsewhere."*

2. **LLaMA 3.3**: *"The main functionality of the `logging_handlers.py` file is to configure logging handlers for a distributed computing framework, specifically by adding a custom logger named 
\"dcp_logger\" to the existing `_log_handlers` dictionary. This logger is initialized with a 
`NullHandler`, which means that it will not output any log messages by default. 
The file updates the `_log_handlers` dictionary from the `torch.distributed.logging_handlers` module, 
allowing for customization of logging behavior in the application."*

3. **Claude 3.5**: *"The file `logging_handlers.py` is primarily focused on setting up logging configurations for a distributed computing environment. It imports the necessary logging modules, defines a constant for a specific logger name (`DCP_LOGGER_NAME`), and updates the _log_handlers dictionary with a NullHandler for the defined logger. This file appears to be part of a larger system for managing logging in a distributed computing context, possibly related to PyTorch's distributed computing functionality."*

4. **GPT-4**: *"The file `logging_handlers.py` is primarily involved in configuring and managing logging handlers for distributed computing processes, specifically within a Python environment using the `logging` module. It imports and updates logging handlers from `torch.distributed.logging_handlers` to specify a `NullHandler` for a logger named 'dcp_logger'. This configuration indicates that the logger for distributed computing processes will intentionally avoid processing log messages by default, until further configuration is potentially applied."*


### A Few Insights
1. Claude 3.5 is 8× more expensive than LLaMA 3.3—but does it deliver 8× more value?
2. GPT-4 is 4× more expensive than Gemini 2.5. However, Gemini’s summaries appear slightly more specific and deterministic.