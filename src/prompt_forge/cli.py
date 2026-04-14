import typer
import os
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from openai import OpenAI

app = typer.Typer(
    name="prompt-forge",
    help="🚀 Python CLI 提示词优化 & 批量测试工具（Grok 4.20 驱动）",
    add_completion=False,
)
console = Console()

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

def get_optimization_system_prompt() -> str:
    return """You are a world-class Prompt Engineer. Improve the user's prompt to be:
1. Clearer and more structured
2. Include specific role, task, and output format requirements
3. Add Chain-of-Thought or Few-shot examples if appropriate
4. Reduce ambiguity and improve LLM output quality
Output only the improved prompt, then separate with --- and add a short "Improvement notes" (3-5 points)."""

@app.command()
def optimize(
    prompt: str = typer.Argument(..., help="你要优化的原始提示词"),
    model: str = typer.Option("grok-4.20-reasoning", "--model", "-m", help="使用的 Grok 模型"),
):
    """一键优化提示词"""
    if not os.getenv("XAI_API_KEY"):
        console.print("[red]错误：请先设置环境变量 XAI_API_KEY[/red]")
        raise typer.Exit(1)

    with console.status("正在调用 Grok 优化提示词..."):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": get_optimization_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
    
    result = response.choices[0].message.content
    console.print(Panel(result, title="✅ Grok 优化后的提示词", border_style="green"))

@app.command()
def batch(
    prompt: str = typer.Argument(..., help="基础提示词"),
    num: int = typer.Option(5, "--num", "-n", help="生成多少个变体", min=3, max=15),
    output: str = typer.Option("prompt_variants.json", "--output", "-o"),
    model: str = typer.Option("grok-4.20-reasoning", "--model", "-m"),
):
    """批量生成测试变体并保存"""
    if not os.getenv("XAI_API_KEY"):
        console.print("[red]错误：请先设置环境变量 XAI_API_KEY[/red]")
        raise typer.Exit(1)

    with console.status(f"正在用 Grok 生成 {num} 个提示词变体..."):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": f"你是一位 Prompt Engineer。请基于下面的提示词，生成 {num} 个高质量的不同变体。每个变体都应保持核心意图，但从不同角度优化。直接输出 JSON 数组，每个元素包含 'id'、'variant'、'reason' 三个字段。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
        )
    
    try:
        variants = json.loads(response.choices[0].message.content)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(variants, f, ensure_ascii=False, indent=2)
        
        table = Table(title=f"📊 Grok 生成的 {num} 个提示词变体")
        table.add_column("ID", style="cyan")
        table.add_column("变体预览", style="green")
        table.add_column("优化理由", style="yellow")
        for v in variants[:5]:
            table.add_row(str(v.get("id")), v.get("variant")[:80] + "...", v.get("reason"))
        console.print(table)
        console.print(f"[bold green]✅ 已保存到 {output}[/bold green]")
    except:
        console.print("[red]JSON 解析失败，已打印原始结果[/red]")
        console.print(response.choices[0].message.content)

if __name__ == "__main__":
    app()
