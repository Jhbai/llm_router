import os
import sys
import json
import requests
import subprocess
import asyncio
import re
import uuid

SYS_PROMPT = r"""
You are Cline, an AI coding agent. Your primary goal is to assist users with various coding tasks by leveraging your knowledge and the tools at your disposal. Given the user's prompt, you should use the tools available to you to answer user's question.

Always gather all the necessary context before starting to work on a task. For example, if you are generating a unit test or new code, make sure you understand the requirement, the naming conventions, frameworks and libraries used and aligned in the current codebase, and the environment and commands used to run and test the code etc. Always validate the new unit test at the end including running the code if possible for live feedback.
Review each question carefully and answer it with detailed, accurate information.
If you need more information, use one of the available tools or ask for clarification instead of making assumptions or lies.

Environment you are running in:
<env>
1. Platform: win32
2. Date: 9/20/2026
3. IDE: VS Code
4. Working Directory: c:\Users\user\Desktop\router
</env>

Remember:
- Always adhere to existing code conventions and patterns.
- Use only libraries and frameworks that are confirmed to be in use in the current codebase.
- Provide complete and functional code without omissions or placeholders.
- Be explicit about any assumptions or limitations in your solution.
- Always show your planning process before executing any task. This will help ensure that you have a clear understanding of the requirements and that your approach aligns with the user's needs.
- Always use absolute paths when referring to files.
- You can call multiple tools in a single response. Before using tools, identify every independent read, search, command, or edit needed for the next step and emit all of those tool calls now, either as multiple tool calls or as one batched input for tools that accept arrays. Do not wait for one independent result before requesting another. Do not split independent reads, searches, checks, or edits across separate turns.
- Good parallelism examples: read all known relevant files in one read_files call; run independent inspection commands in one run_commands call; emit independent read_files, search_codebase, and run_commands calls together in one response; emit multiple editor calls together when editing different files or non-overlapping regions.
- Always verify the files you have edited or created at the end of the task to ensure they are completed and working as expected.

Begin by analyzing the user's input and gathering any necessary additional context. Then, present your plan at the start of your response along with tool calls before proceeding with the task. It's OK for this section to be quite long.

REMEMBER, be helpful and proactive! Don't ask for permission to do something when you can do it! Do not indicates you will be using a tool unless you are actually going to use it.

IMPORTANT: Always includes tool calls in your response until the task is completed. Response without tool calls will considered as completed with final answer.

When you have completed the task, please provide a summary of what you did and any relevant information that the user should know. This will help ensure that the user understands the changes made and can easily follow up if they have any questions or need further assistance. Do not indicate that you will perform an action without actually doing it. Always provide the final result in your response. Always validate your answer with checking the code and running it if possible. 

If user asked a simple question without any coding context, answer it directly without using any tools.
# Plan / Act Modes

User messages arrive wrapped in a <user_input mode="..."> tag. The mode attribute is the interaction mode the user was in when they sent that message: "plan" means plan-mode constraints applied (explore, analyze, and align on a plan -- no edits or state-changing commands), while "act" (or "yolo") means implementation was allowed. If the mode attribute changes between messages, the user switched modes -- the newest message's mode is what governs right now, regardless of what earlier messages allowed. A <mode_notice> block inside a message marks exactly when such a switch happened.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_files",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
                            "required": ["path"]
                        }
                    }
                },
                "required": ["files"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_to_file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "new_text": {"type": "string"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_commands",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}},
                "required": ["command"]
            }
        }
    }
]

def _parse_cline_xml_tools(content: str) -> list:
    calls = []
    for m in re.finditer(r'<invoke\s+name="([^"]+)">([\s\S]*?)</invoke>', content):
        name = m.group(1)
        args = {}
        for pm in re.finditer(r'<parameter\s+name="([^"]+)">([\s\S]*?)</parameter>', m.group(2)):
            args[pm.group(1)] = pm.group(2).strip()
        
        if name == "editor" and args.get("command") == "write_file":
            mapped_name = "write_to_file"
            mapped_args = {"path": args.get("path", ""), "content": args.get("content", "")}
        elif name == "run_commands":
            mapped_name = "run_commands"
            cmd = args.get("commands", "")
            if cmd.startswith("[") and cmd.endswith("]"):
                try:
                    cmd = " && ".join(json.loads(cmd))
                except Exception:
                    pass
            mapped_args = {"command": cmd}
        else:
            mapped_name = name
            mapped_args = args

        calls.append({
            "id": f"call_{uuid.uuid4().hex[:10]}",
            "type": "function",
            "function": {"name": mapped_name, "arguments": json.dumps(mapped_args, ensure_ascii=False)}
        })
    return calls

def _sync_call_llm_stream(url: str, token: str, messages: list) -> dict:
    payload = {"model": "gemini-3.8-flash", "messages": messages, "stream": True, "tools": TOOLS}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    with requests.post(url, json=payload, headers=headers, stream=True) as resp:
        if resp.status_code != 200:
            print(f"\n[HTTP Error] {resp.status_code}: {resp.text}")
            return {"role": "assistant", "content": f"[Error] HTTP {resp.status_code}", "tool_calls": None}
            
        full_content = ""
        t_calls = {}
        
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    data = json.loads(line[6:])
                    if "error" in data:
                        print(f"\n[Server Error]: {data['error']}")
                        continue
                        
                    delta = data["choices"][0].get("delta", {})
                    
                    if "content" in delta and delta["content"]:
                        sys.stdout.write(delta["content"])
                        sys.stdout.flush()
                        full_content += delta["content"]
                        
                    if "tool_calls" in delta:
                        for tc in delta["tool_calls"]:
                            idx = tc["index"]
                            if idx not in t_calls:
                                t_calls[idx] = {"id": tc.get("id"), "type": "function", "function": {"name": tc["function"].get("name", ""), "arguments": ""}}
                            if "function" in tc and "arguments" in tc["function"]:
                                t_calls[idx]["function"]["arguments"] += tc["function"]["arguments"]
                except (json.JSONDecodeError, KeyError):
                    pass
        if full_content:
            print()
            
    res_msg = {"role": "assistant", "content": full_content}
    if t_calls:
        res_msg["tool_calls"] = list(t_calls.values())
    else:
        xml_calls = _parse_cline_xml_tools(full_content)
        if xml_calls:
            res_msg["tool_calls"] = xml_calls
            
    return res_msg

def _sync_execute_tool_calls(tool_calls: list) -> list:
    results = []
    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"])
            if name == "read_files":
                res = []
                for f in args.get("files", []):
                    path = os.path.abspath(f["path"])
                    with open(path, "r", encoding="utf-8") as file:
                        lines = file.readlines()
                        start = max(1, f.get("start_line", 1)) - 1
                        end = f.get("end_line", len(lines))
                        res.append({"path": f["path"], "content": "".join(lines[start:end])})
                out = {"files": res}
            elif name == "write_to_file":
                path = os.path.abspath(args["path"])
                os.makedirs(os.path.dirname(path), exist_ok=True)
                content = args.get("content", args.get("new_text", ""))
                with open(path, "w", encoding="utf-8") as file:
                    file.write(content)
                out = {"status": "success", "bytes": len(content.encode('utf-8'))}
            elif name == "run_commands":
                proc = subprocess.run(
                    args["command"], shell=True, capture_output=True, text=True, 
                    timeout=args.get("timeout", 30000)/1000.0
                )
                out = {"stdout": proc.stdout, "stderr": proc.stderr, "exitCode": proc.returncode}
            else:
                out = {"error": f"Unknown tool: {name}"}
        except Exception as e:
            out = {"error": str(e)}
            
        results.append({
            "tool_call_id": tc["id"],
            "role": "tool",
            "name": name,
            "content": json.dumps(out, ensure_ascii=False)
        })
    return results

async def agent_loop(url: str, token: str, messages: list, max_steps: int = 8):
    for step in range(1, max_steps + 1):
        print(f"\n--- Agent Step {step}/{max_steps} ---")
        
        assistant_msg = await asyncio.to_thread(_sync_call_llm_stream, url, token, messages)
        messages.append(assistant_msg)
        
        if not assistant_msg.get("tool_calls"):
            print("\n[Termination] 任務已完成 (未呼叫工具)")
            break
            
        print("\n[Pending Tools]:", [tc["function"]["name"] for tc in assistant_msg["tool_calls"]])
        approval = await asyncio.to_thread(input, "允許執行上述工具嗎？(Y/n): ")
        if approval.strip().lower() == 'n':
            print("\n[Termination] 使用者拒絕授權，循環終止。")
            messages.append({"role": "user", "content": "The user denied the tool execution. Task aborted."})
            break
            
        tool_results = await asyncio.to_thread(_sync_execute_tool_calls, assistant_msg["tool_calls"])
        messages.extend(tool_results)
    else:
        print(f"\n[Termination] 達到安全執行步數上限 ({max_steps} 步)，強制退出。")
        
    return messages

async def test_main():
    test_url = "http://127.0.0.1:8000/v1/chat/completions"
    test_token = "MyTokenHere"
    initial_messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": "Please run the command 'echo Hello World' and also write 'Testing 123' to a file named 'test_output.txt'."}
    ]
    
    try:
        final_history = await agent_loop(test_url, test_token, initial_messages)
        print("\n\n=== 最終對話歷史 ===")
        print(final_history)
    except Exception as e:
        print(f"\n執行失敗: {e}")

if __name__ == "__main__":
    asyncio.run(test_main())
