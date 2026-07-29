from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from pathlib import Path
import importlib.util
from langchain_core.messages import HumanMessage
import logging 

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Global gateway threshold


root_dir= Path(__file__).resolve().parent.parent.parent # ../../ --> backend
logger.info(root_dir)
backend = FilesystemBackend(root_dir=root_dir, virtual_mode=True)

SKILLS_DIR = Path(root_dir) / "skills"

def _load_scripts(skill_name: str):
    spec = importlib.util.spec_from_file_location(
        f"skills.{skill_name}.scripts.tools", SKILLS_DIR / skill_name / "scripts" / "tools.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_skill(skill_name: str, task: str, *, model, recursion_limit: int = 25, **vars) -> dict:
    """Run one task through a named skill via a deep agent.

    `task` is the per-call ask (page text, conversation, character
    reference, etc.) — plain content, not templated into SKILL.md.
    `**vars` binds this call's sink tools via
    skills/<skill_name>/scripts/tools.py:make_tools(updates, **vars), same
    tool-as-sink convention as today's make_x_tool(updates, ...).
    """
    scripts = _load_scripts(skill_name)
    updates: dict = {}
    tools = scripts.make_tools(updates, **vars)

    agent = create_deep_agent(
        model=model, tools=tools,
        skills=["/skills/"],   # virtual_mode=True uses leading-slash = root_dir-relative virtual root
        system_prompt=(
            f"Use exactly the '{skill_name}' skill for this task. "
            "Do not read, search, list, or glob any files — everything you "
            "need is already in this task message and the skill instructions. "
            "Call `generate_image` directly."
        ),
        backend=backend,
    )
    agent.invoke(
        {"messages": [HumanMessage(content=task)]},
        config={"recursion_limit": recursion_limit},
    )
    return updates
