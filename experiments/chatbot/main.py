from dataclasses import dataclass
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import sys
import numpy as np
import tiktoken

# System path routing for mtorch framework
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Actual Framework Imports
from mtorch.config import Device, set_device, no_grad, to_cpu
from mtorch import Tensor
from mtorch.nn import CausalTransformer
from mtorch.utils.saves import load_model as mtorch_load_model


@dataclass
class Config:
    model_dir: Path = Path("./model")
    model_name: str = "best_chat_model.pkl"
    device: str = "cuda"
    seq_len: int = 256
    max_gen_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 10

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_name


config = Config()
console = Console()

# Set framework execution device (CUDA/CPU fallback)
set_device(config.device)

# Initialize Tokenizer & Vocab Constants
enc = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = enc.n_vocab
EOT_TOKEN = enc.eot_token

# Global placeholders for runtime assets
model = None

# ==========================================
# ASCII ART ASSETS
# ==========================================

BANNER = r"""[bold magenta]
 _____   ____   ___  _____ 
|_   _| | __ ) / _ \_   _|
  | |   |  _ \| | | || |  
  | |   | |_) | |_| || |  
  |_|   |____/ \___/ |_|  
                           
[dim cyan]v1.0.0 | mtorch autograd core engine loaded...[/dim cyan]
[/bold magenta]"""

USER_PROMPT = "[bold cyan]/// USER INPUT \\\\[/bold cyan]\n> "

# ==========================================
# CORE FUNCTIONS
# ==========================================


def print_error(message: str):
    console.print(f"\n[bold red]\[!] ERROR:[/bold red] {message}\n")


def init_system():
    """Initializes the model architecture and maps the compiled weights."""
    global model
    console.print(BANNER)

    with console.status(
        f"[bold blue]\[*] Scanning filesystem path '{config.model_dir}'...[/bold blue]",
        spinner="bouncingBar",
    ):
        # Instantiate your exact 54M Parameter Storytelling Brain
        model = CausalTransformer(
            vocab_size=VOCAB_SIZE,
            d_model=256,
            num_heads=8,
            num_layers=4,
            max_seq_len=8192,
        )

        if config.model_path.exists():
            try:
                # Load weights using mtorch's internal state dict unpacker
                mtorch_load_model(model, str(config.model_path))
                model.eval()  # Freeze training layers / set evaluation context

                console.print(
                    f"[bold green]\[+] SUCCESS:[/bold green] Active weights mapped: [yellow]{config.model_name}[/yellow] -> {config.device.upper()}.\n"
                )
                console.print(
                    "[dim]Type 'exit' or 'quit' to terminate the session.[/dim]\n"
                )
            except Exception as e:
                print_error(f"Failed to map weights to network layers: {str(e)}")
                sys.exit(1)
        else:
            print_error(
                f"Binary asset missing at path: [yellow]{config.model_path}[/yellow]"
            )
            sys.exit(1)


def generate_response(user_input: str) -> str:
    """Runs a complete non-streaming inference forward pass across the framework."""
    global model

    prompt = f"User: {user_input}\nAssistant:"
    context = enc.encode(prompt, allowed_special="all")
    generated_tokens = []

    with no_grad():
        for _ in range(config.max_gen_tokens):
            x_crop = context[-config.seq_len :]
            x_array = np.array([x_crop], dtype=np.int32)

            x_tensor = Tensor(Device.xp.asarray(x_array))

            logits = model(x_tensor)
            last_logits = to_cpu(logits.data)[0, -1, :]

            for token_idx in set(context[-30:]):
                if last_logits[token_idx] > 0:
                    last_logits[token_idx] /= 1.15
                else:
                    last_logits[token_idx] *= 1.15

            last_logits = last_logits / config.temperature

            cutoff = np.sort(last_logits)[-config.top_k]
            last_logits[last_logits < cutoff] = -float("inf")

            last_logits -= np.max(last_logits)
            probs = np.exp(last_logits)
            probs = probs / np.sum(probs)

            next_token = int(np.random.choice(len(probs), p=probs))

            if next_token == EOT_TOKEN:
                break

            context.append(next_token)
            generated_tokens.append(next_token)

            del logits, x_tensor

    if not generated_tokens:
        return "[dim italic]System returned an empty sequence.[/dim italic]"

    return enc.decode(generated_tokens)


def chat_loop():
    while True:
        try:
            user_input = console.input(USER_PROMPT)
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input.strip():
            continue

        if user_input.lower() in ["exit", "quit"]:
            console.print("\n[bold yellow]\[!]Goodbye.[/bold yellow]\n")
            break

        with console.status(
            "[bold magenta]\[#] T-BOT is thinking...[/bold magenta]", spinner="point"
        ):
            try:
                response = generate_response(user_input)
            except Exception as e:
                print_error(f"Inference execution failed: {str(e)}")
                continue

        console.print(
            Panel(
                f"[green]{response.strip()}[/green]",
                title="[bold magenta]\\\\ SYSTEM OUTPUT //[/bold magenta]",
                border_style="magenta",
                padding=(1, 2),
            )
        )
        console.print()


if __name__ == "__main__":
    init_system()
    chat_loop()
