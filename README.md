With all the hype around LLMs promising 10x productivity, I wanted to take a more grounded look at what’s actually being delivered. Benchmarks are everywhere, but like electric vehicle mileage stats, they rarely reflect real-world outcomes.

ROI (Return on Investment) is a straightforward concept: what you gain compared to what you spend. This repository provides a set of benchmarks to evaluate the ROI of large language models.

### What We Do
We randomly selected 100 Python files from the PyTorch repository and tasked approximately 10 different LLMs with summarizing each file in under three sentences. The resulting summaries are publicly available in this repository [Code Comprehension Summaries](https://raw.githubusercontent.com/strudelbots/strudel-llm-benchmarks/refs/heads/shorten-readme/results/pytorch_DB.json).

Using these summaries (and their metadat) we can now start to compare ROI in particl practical metrics like verbosity, latency, cost, accuracy (from a human perspective), and information gain. We can assess how these models actually perform and what that says about their ROI

### How to Contribute
1. Don’t buy me a coffee.
2. Instead, use that $5 to contribute new file summaries to the results [How It Works](/How It Works/).
3. Remember this repo is always under construction. f you find an issue please fix it. 

### How It Works
The main script (`run_file_summarizer.py`) selects files from a repository and submits them to your chosen LLM(s) for analysis.

To run the script, ensure all required environment variables are set (refer to `__init__.py` in the `benchmark_code` directory for details). You may also need to update some hardcoded access points to integrate your models. Additionally, a caching mechanism is implemented to avoid redundant LLM calls.

#### Example: *Core Functionality* Summarized by Three Models for a Single PyTorch File

File: `/torch/distributed/checkpoint/logging_handlers.py`:  
1. **GPT-4o**: *"The provided Python file initializes and configures a logging system by integrating a custom logger named \"dcp_logger\" into PyTorch's distributed logging handlers. It assigns a `NullHandler` to \"dcp_logger\" to ensure log messages with this logger are ignored unless explicitly handled elsewhere. The file appears to set up a placeholder logger for distributed computing purposes without introducing any additional functionality."*,  
2. **GPT-3.5**: *"The provided Python file defines a module for handling logging in a distributed environment with the torch library. It initializes a logging handler for a specific logger named \"dcp_logger\" using a NullHandler to discard all log messages. The file also updates the existing log handlers with the newly defined handler."*  
3. **GPT-4**: *"The Python file named \"logging_handlers.py\" primarily configures logging handlers for a specific logger named \"dcp_logger\". It imports the logging module and a specific function or module named _log_handlers from the torch.distributed.logging_handlers package. The main functionality is updating the _log_handlers dictionary to set a 'NullHandler' for the 'dcp_logger', effectively ensuring that any logs sent to this logger will be ignored or not outputted anywhere by default.".*

ThisGPT-4 is by far the most expensive model in this comparison—costing 4 times more than GPT-4o and 6 
t####BThe script generates JSON files, each containing a summary of a single file for each requested model. At the end of the process, a consolidated JSON file is produced, combining all summaries from all models (e.g., `04-01-2025__gpt-35-turbo_gpt-4_gpt-4o__pytorch2_summary.json`).E