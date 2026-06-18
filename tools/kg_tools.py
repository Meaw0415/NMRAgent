import os
import subprocess
import json
from .decorator import tool

# Configuration
from .path_utils import artifact_path

KG_ROOT = os.environ.get("NMR_KG_DATA_DIR", str(artifact_path("kg")))
CSV_PATH = os.path.join(KG_ROOT, "10m_elementkg_release.csv")
JSON_PATH = os.path.join(KG_ROOT, "knowlege_description_20w.json")

# Global cache for the JSON data to avoid reloading 1.8GB file
_KG_DATA_CACHE = None

@tool(name="kg_search_triples", description="Search the Knowledge Graph (CSV) for triples containing the query string.")
def kg_search_triples(query: str):
    """
    Search the Knowledge Graph (CSV) for triples containing the query string.
    This helps find structured relationships like "Molecule X HAS_PROPERTY Y".
    
    Args:
        query (str): The keyword to search for (e.g., "Benzene", "Melting Point").
    """
    if not os.path.exists(CSV_PATH):
        return f"Error: KG CSV file not found at {CSV_PATH}"
        
    try:
        # Use grep for fast search in large CSV
        # -i: case insensitive
        # -m 20: limit to 20 matches
        cmd = ['grep', '-i', '-m', '20', query, CSV_PATH]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        lines = result.stdout.strip().split('\n')
        if not lines or lines == ['']:
            return f"No triples found matching '{query}'"
            
        # Format the output for readability
        formatted_output = ["Found relevant triples:"]
        for line in lines:
            # CSV: head_type,head_value,relation,tail_type,tail_value
            parts = line.split(',')
            if len(parts) >= 5:
                # Clean up values (remove quotes if any)
                head = parts[1].strip()
                rel = parts[2].strip()
                tail = parts[4].strip()
                formatted_output.append(f"- {head} --[{rel}]--> {tail}")
            else:
                formatted_output.append(f"- {line}")
                
        return "\n".join(formatted_output)
        
    except Exception as e:
        return f"Error searching KG: {str(e)}"

@tool(name="kg_get_description", description="Get detailed description of a chemical entity from the Knowledge Graph corpus.")
def kg_get_description(entity_name: str):
    """
    Get detailed description of a chemical entity from the Knowledge Graph corpus.
    
    Args:
        entity_name (str): The name or SMILES of the entity to look up.
    """
    global _KG_DATA_CACHE
    
    if not os.path.exists(JSON_PATH):
        return f"Error: KG JSON file not found at {JSON_PATH}"
        
    try:
        if _KG_DATA_CACHE is None:
            # Load JSON (this might be slow for large files, but for demo it's fine)
            # For production, we should use a more efficient index or database
            print(f"[System] Loading Knowledge Graph Corpus from {JSON_PATH} (1.8GB)...")
            try:
                with open(JSON_PATH, 'r') as f:
                    _KG_DATA_CACHE = json.load(f)
                print("[System] KG Corpus loaded.")
            except FileNotFoundError:
                return f"Error: KG JSON file not found at {JSON_PATH}"
            except json.JSONDecodeError:
                return f"Error: Invalid JSON format in {JSON_PATH}"
            
        # Optimize search if cache is list of dicts
        if isinstance(_KG_DATA_CACHE, list):
            for item in _KG_DATA_CACHE:
                if entity_name.lower() in item.get('entity', '').lower():
                    # Extract the first paragraph or key info to keep it concise
                    corpus = item['corpus'][0] if isinstance(item['corpus'], list) and item['corpus'] else str(item['corpus'])
                    return f"Description for '{item['entity']}':\n{corpus[:2000]}..." # Truncate if too long
        else:
            return "Error: Unexpected KG data format (expected list of dicts)."
                
        return f"No description found for '{entity_name}'"
        
    except Exception as e:
        return f"Error reading KG JSON: {str(e)}"