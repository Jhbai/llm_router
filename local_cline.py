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
class AgentRuntimeAbortError(Exception): pass
class ControlledStopError(Exception): pass
class ProviderError(Exception): pass

def _parse_cline_xml_tools(content: str) -> list:
    calls = []
    for m in re.finditer(r'<invoke\s+name="([^"]+)">([\s\S]*?)</invoke>', content):
        name = m.group(1)
        args = {}
        for pm in re.finditer(r'<parameter\s+name="([^"]+)">([\s\S]*?)</parameter>', m.group(2)):
            args[pm.group(1)] = pm.group(2).strip()
        
        mapped_name = "write_to_file" if name == "editor" and args.get("command") == "write_file" else name
        mapped_args = {"path": args.get("path", ""), "content": args.get("content", "")} if mapped_name == "write_to_file" else args
        if mapped_name == "run_commands":
            cmd = args.get("commands", "")
            mapped_args = {"command": " && ".join(json.loads(cmd)) if cmd.startswith("[") else cmd}
            
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:10]}",
            "type": "function",
            "function": {"name": mapped_name, "arguments": json.dumps(mapped_args, ensure_ascii=False)}
        })
    return calls

def _sync_call_llm_stream(url: str, token: str, messages: list):
    payload = {"model": "gemini-3.8-flash", "messages": messages, "stream": True, "tools": TOOLS}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    with requests.post(url, json=payload, headers=headers, stream=True, timeout=30) as resp:
        if resp.status_code != 200:
            raise ProviderError(f"HTTP {resp.status_code}: {resp.text}")
            
        full_content, t_calls, finish_reason = "", {}, None
        
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("data: ") and line != "data: [DONE]":
                try:
                    data = json.loads(line[6:])
                    if "error" in data:
                        raise ProviderError(data['error'])
                        
                    choice = data["choices"][0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason") or finish_reason
                    
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
    tools = list(t_calls.values()) if t_calls else _parse_cline_xml_tools(full_content)
    if tools:
        res_msg["tool_calls"] = tools
    return res_msg, finish_reason

async def safe_call_llm(url: str, token: str, messages: list, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return await asyncio.to_thread(_sync_call_llm_stream, url, token, messages)
        except (requests.RequestException, ProviderError) as e:
            if attempt == max_retries - 1:
                raise e
            await asyncio.sleep(2 ** attempt)

async def agent_loop(url: str, token: str, messages: list, config: dict = None):
    config = config or {"maxIterations": 8, "maxConsecutiveMistakes": 6}
    abort_event = asyncio.Event() 
    
    consecutive_mistakes = 0
    reminders_issued = 0
    previous_tool_signature = None
    repeated_tool_count = 0

    # 1. Lifecycle Hook: beforeRun
    print("[Hook] beforeRun invoked")
    
    for step in range(1, config["maxIterations"] + 1):
        if abort_event.is_set():
            raise AgentRuntimeAbortError("Run aborted by user/system signal.")
            
        print(f"\n--- Agent Step {step}/{config['maxIterations']} ---")
        
        try:
            assistant_msg, finish_reason = await safe_call_llm(url, token, messages)
        except Exception as e:
            raise ProviderError(f"Provider Retries Exhausted: {str(e)}")

        # 2. Model Streaming Termination Criteria Evaluation
        content_empty = not assistant_msg.get("content", "").strip()
        has_tools = bool(assistant_msg.get("tool_calls"))

        if finish_reason == "aborted":
            raise AgentRuntimeAbortError("Stream aborted by provider.")
        elif content_empty and not has_tools:
            if finish_reason == "error":
                raise ProviderError("Provider error with empty response.")
            raise ValueError("Model returned empty response without tool calls.")
        elif finish_reason == "max_tokens" and not has_tools:
            raise RuntimeError("Fatal: Max context limit reached before tool emission.")

        messages.append(assistant_msg)
        
        # 3. Loop Progression & Goal Completion
        if not has_tools:
            if reminders_issued < 1:
                reminders_issued += 1
                messages.append({"role": "user", "content": "Please invoke tools or finalize the answer using attempt_completion."})
                continue
            else:
                print("\n[Termination] Explicit finalization skipped; Run completed implicitly.")
                break

        # 4. Guardrails: Repetitive Tool Detection
        current_sig = hashlib.md5(json.dumps([tc["function"] for tc in assistant_msg["tool_calls"]], sort_keys=True).encode()).hexdigest()
        if current_sig == previous_tool_signature:
            repeated_tool_count += 1
            if repeated_tool_count == 2:
                messages.append({"role": "user", "content": "You are repeating the same action. Please try an alternate approach."})
                continue
            elif repeated_tool_count > 2:
                consecutive_mistakes += 1
        else:
            previous_tool_signature = current_sig
            repeated_tool_count = 0

        # 5. Tool Approval & Human Intervention
        print("\n[Pending Tools]:", [tc["function"]["name"] for tc in assistant_msg["tool_calls"]])
        approval = await asyncio.to_thread(input, "允許執行上述工具嗎？(Y/n): ")
        
        if approval.strip().lower() == 'n':
            rejection_results = [{
                "tool_call_id": tc["id"],
                "role": "tool",
                "name": tc["function"]["name"],
                "content": json.dumps({"error": "User denied authorization."})
            } for tc in assistant_msg["tool_calls"]]
            messages.extend(rejection_results)
            consecutive_mistakes += 1
            continue

        # Execute Tools
        tool_results = []
        all_failed = True
        run_completed = False
        
        for tc in assistant_msg["tool_calls"]:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
                if name == "attempt_completion":
                    out = {"status": "success", "result": args.get("result", "")}
                    run_completed = True
                    all_failed = False
                elif name == "read_files":
                    res = [{"path": f["path"], "content": open(f["path"], "r", encoding="utf-8").read()} for f in args.get("files", [])]
                    out = {"files": res}
                    all_failed = False
                elif name == "write_to_file":
                    os.makedirs(os.path.dirname(args["path"]), exist_ok=True)
                    with open(args["path"], "w", encoding="utf-8") as f:
                        f.write(args.get("content", ""))
                    out = {"status": "success"}
                    all_failed = False
                elif name == "run_commands":
                    proc = subprocess.run(args["command"], shell=True, capture_output=True, text=True, timeout=args.get("timeout", 30000)/1000.0)
                    out = {"stdout": proc.stdout, "stderr": proc.stderr, "exitCode": proc.returncode}
                    all_failed = False
                else:
                    out = {"error": f"Unknown tool: {name}"}
            except Exception as e:
                out = {"error": str(e)}
                
            tool_results.append({
                "tool_call_id": tc["id"],
                "role": "tool",
                "name": name,
                "content": json.dumps(out, ensure_ascii=False)
            })

        messages.extend(tool_results)
        consecutive_mistakes = consecutive_mistakes + 1 if all_failed else 0

        # Max Mistake Guardrail
        if consecutive_mistakes >= config["maxConsecutiveMistakes"]:
            raise ControlledStopError("Max consecutive mistakes limit reached. Aborting run.")

        # Explicit Terminal Tool completed run
        if run_completed:
            print("\n[Termination] Run Completed Explicitly via attempt_completion.")
            break
            
    else:
        raise RuntimeError(f"Max iterations ({config['maxIterations']}) reached.")
        
    print("[Hook] afterRun invoked: run-finished")
    return messages

async def test_main():
    test_url = "http://127.0.0.1:8000/v1/chat/completions"
    test_token = "MyTokenHere"
    initial_messages = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": "write a hello world python code in ./test/demo.py"}
    ]
    
    try:
        final_history = await agent_loop(test_url, test_token, initial_messages)
        print(f"\n=== 最終對話歷史 ===")
        print(final_history)
    except Exception as e:
        print(f"\n執行終止: {type(e).__name__} - {e}")

if __name__ == "__main__":
    asyncio.run(test_main())
