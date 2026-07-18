import os
import sys
import json
import subprocess
import re
from pathlib import Path

# Force UTF-8 encoding for safety
sys.stdout.reconfigure(encoding='utf-8')

def find_file_recursively(start_path, filename):
    for path in Path(start_path).rglob(filename):
        return path
    return None

def sanitize_rtl(content):
    content = re.sub(r'\$stop\s*\([^)]*\)\s*;', ' ; ', content)
    content = re.sub(r'\$stop\b\s*;', ' ; ', content)
    content = re.sub(r'\$finish\s*\([^)]*\)\s*;', ' ; ', content)
    content = re.sub(r'\$finish\b\s*;', ' ; ', content)
    return content

def process_assertion(prop_code):
    prop_code = re.sub(r"'\s*\(", "(", prop_code)
    prop_code = re.sub(r"(?<!\d)'([a-zA-Z_]\w*)'?", r"\1", prop_code)
    match = re.search(r'(always\s+@.*?\nend)', prop_code, re.DOTALL)
    if match:
        return match.group(1)
    return prop_code

def evaluate_rules_isolated(run_folder_path, top_module="eth_top"):
    run_dir = Path(run_folder_path)
    if run_dir.name == "00_summary":
        json_path = run_dir / "results_summary.json"
        formal_dir = run_dir.parent / "03_formal"
    else:
        json_paths = list(run_dir.rglob("results_summary.json"))
        json_path = json_paths[0] if json_paths else run_dir / "results_summary.json"
        formal_dir = run_dir / "03_formal"
        
    if not json_path.exists():
        print(f"Error: Could not find results_summary.json")
        sys.exit(1)
        
    formal_dir.mkdir(parents=True, exist_ok=True)
    sanitized_dir = formal_dir / "sanitized_rtl"
    sanitized_dir.mkdir(exist_ok=True)
    
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
        
    rtl_base_dir = Path("designs")
    target_rtl_file = find_file_recursively(rtl_base_dir, f"{top_module}.v")
    
    if not target_rtl_file:
        top_module = "ethmac"
        target_rtl_file = find_file_recursively(rtl_base_dir, f"{top_module}.v")

    if not target_rtl_file:
        print(f"Error: Could not find top module {top_module}.v")
        sys.exit(1)
        
    rtl_target_dir = target_rtl_file.parent
    read_cmds = ""
    
    for v_file in rtl_target_dir.glob("*.v"):
        if v_file.name != target_rtl_file.name:
            with open(v_file, 'r', encoding='utf-8') as f:
                content = sanitize_rtl(f.read())
            with open(sanitized_dir / v_file.name, 'w', encoding='utf-8') as f:
                f.write(content)
            read_cmds += f"read_verilog -sv {(sanitized_dir / v_file.name).resolve().as_posix()}\n"

    with open(target_rtl_file, 'r', encoding='utf-8') as f:
        original_top_content = sanitize_rtl(f.read())

    print("\n" + "="*70)
    print(" ISOLATED FORMAL VERIFICATION BENCHMARK")
    print("="*70)

    for rule in results:
        if not rule.get("tier1_passed", False):
            continue
            
        rule_id = rule.get('rule_id', 'Unknown')
        prop_code = rule.get("property_code", "")
        prop_code = prop_code.replace("```verilog", "").replace("```systemverilog", "").replace("```", "").strip()
        clean_code = process_assertion(prop_code)
        
        if "endmodule" in original_top_content:
            injected_content = original_top_content.rsplit("endmodule", 1)[0] + f"\n// --- LLM Rule: {rule_id} ---\n{clean_code}\nendmodule\n"
        else:
            injected_content = original_top_content + f"\n// --- LLM Rule: {rule_id} ---\n{clean_code}\n"
            
        top_safe_file = sanitized_dir / target_rtl_file.name
        with open(top_safe_file, 'w', encoding='utf-8') as f:
            f.write(injected_content)
            
        # UNIQUE SBY FILE FOR EACH RULE
        sby_run_name = f"prove_{rule_id}"
        sby_config = f"""[options]
mode prove
depth 20
wait on

[engines]
smtbmc boolector

[script]
{read_cmds}
read_verilog -sv {top_safe_file.resolve().as_posix()}
prep -top {top_module}
"""
        sby_path = formal_dir / f"{sby_run_name}.sby"
        with open(sby_path, 'w', encoding='utf-8') as f:
            f.write(sby_config)
            
        result = subprocess.run(["sby", "-f", sby_path.name], cwd=formal_dir, capture_output=True, text=True, encoding='utf-8', errors='replace')
        
        print(f"\n[{rule_id}] CODE:")
        print("-" * 50)
        print(clean_code.strip())
        print("-" * 50)

        if "DONE (PASS, rc=0)" in result.stdout:
            print(f"[{rule_id}] RESULT: ✅ PASSED (Mathematically Proved)")
        elif "DONE (FAIL, rc=2)" in result.stdout:
            print(f"[{rule_id}] RESULT: ❌ FAILED (Counter-example found)")
            print("  [Trace Details]:")
            for line in result.stdout.splitlines() + result.stderr.splitlines():
                if "Assert failed" in line or "trace:" in line or "failed" in line.lower() and "engine" in line:
                    print(f"    -> {line.strip().replace('SBY [prove] ', '')}")
            
            # CORRECT UNIQUE VCD PATH
            vcd_path = formal_dir / sby_run_name / "engine_0" / "trace.vcd"
            print(f"    -> Waveform generated: {vcd_path.resolve()}")
            
        else:
            error_msg = "Unknown Elaboration Error"
            for line in result.stdout.splitlines() + result.stderr.splitlines():
                if "ERROR:" in line and "task failed" not in line:
                    error_msg = line.split("ERROR:")[-1].strip()
                    break
            print(f"[{rule_id}] RESULT: ⚠️ CRASHED")
            print(f"  -> {error_msg}")

    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval.py <path_to_run_folder> [top_module_name]")
        print("Example: python eval.py results/i2c/run_123 i2c_master_top")
        sys.exit(1)
        
    run_folder = sys.argv[1]
    # Grab the top module from the terminal, or default to eth_top
    top_mod = sys.argv[2] if len(sys.argv) > 2 else "eth_top"
    
    evaluate_rules_isolated(run_folder, top_module=top_mod)