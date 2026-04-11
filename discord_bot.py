#!/usr/bin/env python3
"""
[선택·테스트용] 프로덕션 API 경로(main.py)와 분리된 보조 클라이언트입니다. 미실행 시 기존 동작에 영향 없음.

팀 디스코드에서 URL을 보내면 로컬 Phishing API(/analyze) 결과를 채팅으로 돌려줍니다.

사전 준비
---------
1. Discord Developer Portal에서 Application 생성 → Bot → Token 복사
2. Bot 권한: Send Messages, Embed Links, Read Message History, Attach Files (analyze_debug.json 첨부)
   (메시지로 URL 받으려면 MESSAGE CONTENT INTENT 켜기 — Privileged Gateway Intent)
3. OAuth2 URL Generator: `bot` + `applications.commands` 스코프, 위 권한 체크 후 초대 링크로 서버에 초대
4. 이 머신에서 API 실행: `python main.py` (기본 http://127.0.0.1:8000)
5. 환경변수: DISCORD_BOT_TOKEN 필수

실행
----
  export DISCORD_BOT_TOKEN='봇토큰'
  export PHISH_API_BASE='http://127.0.0.1:8000'   # 선택
  python discord_bot.py

선택 환경변수
-------------
  PHISH_API_BASE     API 베이스 URL (기본 http://127.0.0.1:8000)
  DISCORD_GUILD_ID   숫자면 슬래시 명령을 해당 길드에만 즉시 동기화(개발용)
  DISCORD_ANALYZE_TIMEOUT_SEC  기본 180
  DISCORD_DEBUG_ATTACH_JSON  기본 1 — 검증 성공 시 전체 API JSON 을 analyze_debug.json 첨부

배포(!최신화) — 봇이 돌아가는 머신에서만 동작 (채널에 접근 가능한 누구나 실행 가능, 소규모 팀 가정)
---------------------------------------------------------------------------
  DISCORD_REPO_ROOT         git / main.py 기준 디렉터리 (기본: 이 스크립트 위치)
  PHISH_API_PORT            main.py 가 listen 하는 포트 (기본 8000, 중지 시 fuser로 해당 포트)
  DISCORD_DEPLOY_API_LOG    재시작한 main.py 로그 파일명 (기본 api_server.log, REPO_ROOT 기준)

  순서: 해당 포트 LISTEN 프로세스 종료 → git fetch → git pull → git lfs pull →
        pip install -r requirements.txt → 같은 인터프리터로 main.py 백그라운드 실행
  주의: 팀원이 봇 채널에 코드를 올려도 git에는 반영되지 않음. pull은 origin만 갱신.
"""
from __future__ import annotations

import asyncio
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass
import io
import json
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


import discord
import httpx
from discord import app_commands
from discord.ext import commands

# --- 설정 ---
PHISH_API_BASE = os.getenv("PHISH_API_BASE", "http://127.0.0.1:8000").rstrip("/")
ANALYZE_PATH = "/analyze"
TIMEOUT_SEC = float(os.getenv("DISCORD_ANALYZE_TIMEOUT_SEC", "180"))
DEBUG_ATTACH_JSON = os.getenv("DISCORD_DEBUG_ATTACH_JSON", "1").strip() not in ("0", "false", "no")
TOKEN = os.getenv("DISCORD_BOT_TOKEN", "").strip()

PREFIXES = ("!검증", "!check", "!url", "!scan", "!피싱")
DEPLOY_COMMANDS = ("!최신화", "!deploy")

REPO_ROOT = os.path.abspath(
    os.getenv("DISCORD_REPO_ROOT", os.path.dirname(os.path.abspath(__file__)))
)
API_LISTEN_PORT = int(os.getenv("PHISH_API_PORT", "8000"))
DEPLOY_API_LOG = os.getenv("DISCORD_DEPLOY_API_LOG", "api_server.log")

deploy_lock = asyncio.Lock()
# Popen 후 부모가 File 핸들을 잃으면 GC가 닫아 자식 stdout 이 깨질 수 있어 유지
_KEEP_DEPLOY_LOG_HANDLES: list[object] = []


def _chop(s: str, limit: int = 900) -> str:
    s = s.strip()
    if len(s) <= limit:
        return s
    return s[: limit - 20] + "\n…(truncated)"


def _sync_deploy_steps() -> List[Tuple[str, int, str]]:
    """포트 중지 → git → lfs → pip → main.py 재실행. 동기 함수(executor에서 호출)."""
    results: List[Tuple[str, int, str]] = []
    env_git = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    def run_step(name: str, cmd: list[str], cwd: str, timeout: int = 600) -> None:
        try:
            p = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env_git,
            )
            combined = (p.stdout or "") + (p.stderr or "")
            results.append((name, p.returncode, _chop(combined, 1200)))
        except subprocess.TimeoutExpired as e:
            results.append((name, -1, f"timeout: {e}"))
        except FileNotFoundError as e:
            results.append((name, -1, f"not found: {e}"))
        except Exception as e:
            results.append((name, -1, str(e)))

    # LISTEN 중인 API 프로세스 종료 (봇 프로세스는 다른 포트를 쓰지 않음)
    try:
        p = subprocess.run(
            ["fuser", "-k", f"{API_LISTEN_PORT}/tcp"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=45,
        )
        combined = (p.stdout or "") + (p.stderr or "")
        results.append(
            (
                f"stop_port_{API_LISTEN_PORT}",
                p.returncode,
                _chop(combined or "(no output; nothing on port?)", 800),
            )
        )
    except FileNotFoundError:
        results.append(
            (
                f"stop_port_{API_LISTEN_PORT}",
                -1,
                "`fuser` 없음. `sudo apt install psmisc` 후 재시도하거나 수동으로 main.py 종료.",
            )
        )

    run_step("git_fetch", ["git", "fetch", "origin"], REPO_ROOT)
    run_step("git_pull", ["git", "pull"], REPO_ROOT)
    run_step("git_lfs_pull", ["git", "lfs", "pull"], REPO_ROOT)
    run_step(
        "pip_install",
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        REPO_ROOT,
    )

    log_path = os.path.join(REPO_ROOT, DEPLOY_API_LOG)
    try:
        logf = open(log_path, "ab")
        _KEEP_DEPLOY_LOG_HANDLES.append(logf)
        subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=REPO_ROOT,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        results.append(
            (
                "start_main_py",
                0,
                f"started `{sys.executable} main.py` (log append → `{log_path}`)",
            )
        )
    except Exception as e:
        results.append(("start_main_py", -1, str(e)))

    return results


async def handle_deploy_command(message: discord.Message) -> None:
    async with deploy_lock:
        status = await message.reply(
            f"배포 시작… (`{REPO_ROOT}`)\n"
            f"포트 `{API_LISTEN_PORT}` 정리 → git → lfs → pip → `main.py` 재실행",
            mention_author=False,
        )
        loop = asyncio.get_running_loop()
        try:
            steps = await loop.run_in_executor(None, _sync_deploy_steps)
        except Exception as e:
            await status.edit(content=f"배포 중 예외: `{e}`")
            return

        lines: list[str] = []
        for name, code, detail in steps:
            mark = "ok" if code == 0 else "check"
            lines.append(f"**{name}** [{mark}] exit={code}\n```{_chop(detail, 500)}```")

        text = "**배포 로그**\n" + "\n".join(lines)
        if len(text) > 1900:
            text = text[:1850] + "\n…(출력 잘림)"
        try:
            await status.edit(content=text)
        except discord.HTTPException:
            await message.channel.send(text[:1999])


_URL_RE = re.compile(
    r"https?://[^\s<>`\[\]()]+|(?:^|\s)([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(?:/[^\s]*)?",
    re.IGNORECASE,
)


def _extract_url(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    m = _URL_RE.search(text)
    if not m:
        return None
    u = m.group(0).strip()
    if not u.startswith("http"):
        u = "https://" + u.lstrip("./")
    return u


def _fmt_block(data: Dict[str, Any]) -> str:
    lines: list[str] = []
    kb = data.get("koBERT") or {}
    if kb.get("engine_disabled"):
        lines.append(f"**KoBERT**: 비활성 ({kb.get('engine_reason', '')[:120]})")
    else:
        j = kb.get("judgment", "?")
        rl = kb.get("riskLevel") or kb.get("risklevel", "?")
        lines.append(f"**KoBERT**: judgment=`{j}`  risk=`{rl}`")

    xg = data.get("xgboost")
    if xg is None:
        lines.append("**XGBoost**: 스킵 (모델 없음)")
    else:
        def _verdict_lbl(lbl: Any) -> str:
            if lbl == 1:
                return "malicious"
            if lbl == 0:
                return "benign"
            return str(lbl)

        sub: list[str] = []
        if "typo_label" in xg or "typo_probability" in xg:
            sub.append(
                f"타이포: `{_verdict_lbl(xg.get('typo_label'))}`  "
                f"p={xg.get('typo_probability')}"
            )
        if "domain_label" in xg or "domain_probability" in xg:
            sub.append(
                f"도메인: `{_verdict_lbl(xg.get('domain_label'))}`  "
                f"p={xg.get('domain_probability')}"
            )
        fp = xg.get("final_probability")
        if sub:
            lines.append("**XGBoost**\n" + "\n".join(f"- {s}" for s in sub))
            lines.append(
                f"- 최종: `{xg.get('verdict')}`  final_p={fp}  (max typo·domain)"
            )
        else:
            lines.append(
                f"**XGBoost**: `{xg.get('verdict')}`  final_p={fp}"
            )

    gnn = data.get("gnn")
    if gnn is None:
        st = data.get("gnn_status") or {}
        lines.append(f"**GNN(lexical)**: 스킵 — `{st.get('reason', '')[:80]}`")
    elif isinstance(gnn, dict) and gnn.get("error"):
        lines.append(f"**GNN(lexical)**: 오류 — `{gnn.get('error', '')[:120]}`")
    else:
        lines.append(
            f"**GNN(lexical)**: `{gnn.get('verdict')}`  p={gnn.get('probability')}"
        )

    t = data.get("duration_sec")
    if t is not None:
        lines.append(f"_소요: {t}s_")
    return "\n".join(lines)


def _json_snippet(obj: Any, limit: int = 980) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        s = repr(obj)
    s = _chop(s.strip(), limit)
    return f"```json\n{s}\n```"


def analyze_debug_attachment(body: Dict[str, Any]) -> discord.File:
    """전체 /analyze 응답 — 디버깅용 다운로드."""
    raw = json.dumps(body, ensure_ascii=False, indent=2, default=str)
    return discord.File(io.BytesIO(raw.encode("utf-8")), filename="analyze_debug.json")


async def call_analyze(client: httpx.AsyncClient, url: str) -> Dict[str, Any]:
    r = await client.post(
        f"{PHISH_API_BASE}{ANALYZE_PATH}",
        json={"url": url},
        timeout=TIMEOUT_SEC,
    )
    r.raise_for_status()
    return r.json()


def build_embed(target_url: str, body: Dict[str, Any], error: Optional[str] = None) -> discord.Embed:
    if error:
        return discord.Embed(
            title="검증 실패",
            description=f"```\n{error[:1800]}\n```",
            color=discord.Color.red(),
        )
    emb = discord.Embed(
        title="피싱 검증 결과 (debug)",
        description=_fmt_block(body)[:3500],
        color=discord.Color.orange(),
    )
    emb.add_field(name="URL", value=f"`{target_url[:1000]}`", inline=False)
    emb.add_field(
        name="API",
        value=f"`{PHISH_API_BASE}{ANALYZE_PATH}`",
        inline=False,
    )
    tim = body.get("timing")
    if tim:
        emb.add_field(
            name="timing (sec)",
            value=_json_snippet(tim, 900),
            inline=False,
        )
    emb.set_footer(
        text="상세(engine/xgboost/gnn status·raw) → analyze_debug.json 첨부 "
        "(DISCORD_DEBUG_ATTACH_JSON=0 이면 첨부 없음)"
    )
    return emb


intents = discord.Intents.default()
intents.message_content = True


class PhishBot(commands.Bot):
    def __init__(self) -> None:
        super().__init__(command_prefix=commands.when_mentioned, intents=intents)
        self.http_analyze: Optional[httpx.AsyncClient] = None

    async def setup_hook(self) -> None:
        self.http_analyze = httpx.AsyncClient()
        guild_id = os.getenv("DISCORD_GUILD_ID")
        if guild_id and guild_id.isdigit():
            g = discord.Object(id=int(guild_id))
            self.tree.copy_global_to(guild=g)
            await self.tree.sync(guild=g)
            print(f"[discord] slash commands synced to guild {guild_id}")
        else:
            await self.tree.sync()
            print("[discord] slash commands synced globally (may take up to 1h to appear)")

    async def close(self) -> None:
        if self.http_analyze:
            await self.http_analyze.aclose()
        await super().close()


bot = PhishBot()


@bot.tree.command(name="phish", description="URL 피싱 검증 (KoBERT / XGBoost / GNN)")
@app_commands.describe(url="검사할 주소 (https://... 또는 도메인)")
async def slash_phish(interaction: discord.Interaction, url: str) -> None:
    target = _extract_url(url) or url.strip()
    if not target:
        await interaction.response.send_message("URL이 비어 있습니다.", ephemeral=True)
        return
    await interaction.response.defer()
    assert bot.http_analyze is not None
    try:
        body = await call_analyze(bot.http_analyze, target)
        emb = build_embed(body.get("url", target), body)
        if DEBUG_ATTACH_JSON:
            await interaction.followup.send(
                embed=emb, file=analyze_debug_attachment(body)
            )
        else:
            await interaction.followup.send(embed=emb)
    except httpx.ConnectError as e:
        await interaction.followup.send(
            f"API에 연결할 수 없습니다. `{PHISH_API_BASE}` 에서 `python main.py` 가 떠 있는지 확인하세요.\n`{e}`"
        )
    except httpx.HTTPStatusError as e:
        await interaction.followup.send(
            embed=build_embed(target, {}, error=f"HTTP {e.response.status_code}: {e.response.text[:500]}")
        )
    except Exception as e:
        await interaction.followup.send(embed=build_embed(target, {}, error=str(e)))


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    raw = message.content.strip()
    if not raw:
        return

    parts = raw.split()
    if parts and parts[0] in DEPLOY_COMMANDS:
        await handle_deploy_command(message)
        return

    rest: Optional[str] = None
    for p in PREFIXES:
        if raw.startswith(p):
            rest = raw[len(p) :].strip()
            break
    if rest is None:
        return

    target = _extract_url(rest) or rest.strip()
    if not target:
        await message.reply("URL을 찾지 못했습니다. 예: `!검증 https://example.com`", mention_author=False)
        return

    assert bot.http_analyze is not None
    async with message.channel.typing():
        try:
            body = await call_analyze(bot.http_analyze, target)
            emb = build_embed(body.get("url", target), body)
            if DEBUG_ATTACH_JSON:
                await message.reply(
                    embed=emb,
                    file=analyze_debug_attachment(body),
                    mention_author=False,
                )
            else:
                await message.reply(embed=emb, mention_author=False)
        except httpx.ConnectError as e:
            await message.reply(
                f"API 연결 실패 (`{PHISH_API_BASE}`). 서버에서 `python main.py` 실행 여부를 확인하세요.\n`{e}`",
                mention_author=False,
            )
        except httpx.HTTPStatusError as e:
            await message.reply(
                embed=build_embed(target, {}, error=f"HTTP {e.response.status_code}"),
                mention_author=False,
            )
        except Exception as e:
            await message.reply(embed=build_embed(target, {}, error=str(e)), mention_author=False)


@bot.event
async def on_ready() -> None:
    print(f"[discord] logged in as {bot.user} (id={bot.user.id if bot.user else '?'})")
    print(f"[discord] API: {PHISH_API_BASE}{ANALYZE_PATH}")
    print(
        f"[discord] deploy (!최신화): repo={REPO_ROOT}, port={API_LISTEN_PORT}"
    )


def main() -> int:
    if not TOKEN:
        print(
            "error: DISCORD_BOT_TOKEN 이 비어 있습니다.\n"
            "  export DISCORD_BOT_TOKEN='...'\n"
            "  python discord_bot.py",
            file=sys.stderr,
        )
        return 1
    bot.run(TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
