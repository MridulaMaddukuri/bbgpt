# bbgpt
Follow along Karpathy's video and create an endpoint

## Getting started with the repository

I'm using uv for package management in this repository

1. Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
2. Add uv to path 
```bash
source $HOME/.cargo/env
```
3. Create a vitual enviroment with your desired version of python(I used 3.12) and activate it
```bash
uv venv --python 3.12
source .venv/bin/activate
```
4. Initialize the project. This command will generate `pyproject.toml` and `.python-version` files
```bash
uv init
```
5. Now add dependencies as needed
```bash
uv add torch numpy tqdm transformers tiktoken wandb datasets pre-commit
```
6. Install pre-commit hooks
```bash
uv run pre-commit install
```



## Notes

- Download tiny shakespeare dataset
`wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`
- To check the cuda version on your machine, try 
```bash
/usr/local/cuda/bin/nvcc --version
or 
nvcc --verison
```
- To format files
`uv run ruff format gpt.py` 