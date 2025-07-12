#generate_config_hpp.py

def write_config_header(config: dict, output_path: str = "config.h"):
    def flatten_dict(d, prefix=""):
        flat = {}
        for k, v in d.items():
            full_key = f"{prefix}_{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_dict(v, full_key))
            else:
                flat[full_key] = v
        return flat
    
    flat_config = flatten_dict(config)


    lines = ["#pragma once", "", "#include <string_view>", "", "namespace config {"]
    for key, val in flat_config.items():
        if isinstance(val, bool):
            cpp_val = "true" if val else "false"
            lines.append(f"\tstatic constexpr bool {key} = {cpp_val};")
        elif isinstance(val, int):
            lines.append(f"\tstatic constexpr int {key} = {val};")
        elif isinstance(val, float):
            lines.append(f"\tstatic constexpr float {key} = {val};")#TODO does this handle doubles?
        elif isinstance(val, str):
            lines.append(f"\tstatic constexpr str {key} = {val};")
        elif isinstance(val, list):
            array_elements = ", ".join(str(x) for x in val)
            lines.append(f"\tconstexpr std::array<int,{len(val)}> {key} = {array_elements};")
        
        else:
            raise ValueError(f"Unsupported config values {key} = {val} (type {type(val)})")
    lines.append("}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))