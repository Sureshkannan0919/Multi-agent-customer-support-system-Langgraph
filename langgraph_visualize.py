#!/usr/bin/env python3
"""
LangGraph Built-in Visualization
This script uses LangGraph's built-in visualization capabilities to display the graph structure.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def visualize_with_langgraph():
    """Visualize the graph using LangGraph's built-in Mermaid visualization."""
    
    try:
        # Import the graph from main.py
        from main import graph
        
        print("🚀 Creating LangGraph visualization...")
        
        # Method 1: Try to get Mermaid PNG
        try:
            print("📊 Generating Mermaid PNG visualization...")
            mermaid_png = graph.get_graph().draw_mermaid_png()
            
            # Save the PNG
            with open("langgraph_mermaid.png", "wb") as f:
                f.write(mermaid_png)
            print("✅ Mermaid PNG saved as: langgraph_mermaid.png")
            
        except Exception as e:
            print(f"⚠️ Mermaid PNG generation failed: {e}")
            print("This might be due to missing dependencies (pygraphviz, etc.)")
        
        # Method 2: Try to get Mermaid text
        try:
            print("📝 Generating Mermaid text representation...")
            mermaid_text = graph.get_graph().draw_mermaid()
            
            # Save the Mermaid text
            with open("langgraph_mermaid.mmd", "w") as f:
                f.write(mermaid_text)
            print("✅ Mermaid text saved as: langgraph_mermaid.mmd")
            
            # Display the Mermaid text
            print("\n" + "="*60)
            print("MERMAID DIAGRAM TEXT:")
            print("="*60)
            print(mermaid_text)
            print("="*60)
            
        except Exception as e:
            print(f"⚠️ Mermaid text generation failed: {e}")
        
        # Method 3: Try to get ASCII representation
        try:
            print("📋 Generating ASCII representation...")
            ascii_repr = graph.get_graph().draw_ascii()
            
            # Save the ASCII
            with open("langgraph_ascii.txt", "w") as f:
                f.write(ascii_repr)
            print("✅ ASCII representation saved as: langgraph_ascii.txt")
            
            # Display the ASCII
            print("\n" + "="*60)
            print("ASCII DIAGRAM:")
            print("="*60)
            print(ascii_repr)
            print("="*60)
            
        except Exception as e:
            print(f"⚠️ ASCII generation failed: {e}")
        
        # Method 4: Get graph info
        try:
            print("ℹ️ Getting graph information...")
            graph_info = graph.get_graph()
            print(f"✅ Graph nodes: {len(graph_info.nodes)}")
            print(f"✅ Graph edges: {len(graph_info.edges)}")
            
            # Print node details
            print("\n📋 NODES:")
            for node_id, node_data in graph_info.nodes.items():
                print(f"   • {node_id}: {node_data}")
            
            # Print edge details
            print("\n🔗 EDGES:")
            for edge in graph_info.edges:
                print(f"   • {edge[0]} → {edge[1]}")
                
        except Exception as e:
            print(f"⚠️ Graph info extraction failed: {e}")
        
        print("\n✅ LangGraph visualization completed!")
        print("📁 Files created:")
        print("   • langgraph_mermaid.png - Mermaid PNG (if successful)")
        print("   • langgraph_mermaid.mmd - Mermaid text")
        print("   • langgraph_ascii.txt - ASCII representation")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requiremts.txt")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("This might be due to missing dependencies or graph compilation issues.")

def visualize_with_ipython():
    """Visualize using IPython display (for Jupyter notebooks)."""
    
    try:
        from IPython.display import Image, display
        from main import graph
        
        print("🔬 Using IPython display for visualization...")
        
        try:
            # Generate and display the Mermaid PNG
            mermaid_png = graph.get_graph().draw_mermaid_png()
            display(Image(mermaid_png))
            print("✅ IPython display successful!")
            
        except Exception as e:
            print(f"⚠️ IPython display failed: {e}")
            print("This is expected if not running in a Jupyter environment.")
            
    except ImportError:
        print("⚠️ IPython not available - this method only works in Jupyter notebooks")
    except Exception as e:
        print(f"❌ Error with IPython visualization: {e}")

def main():
    """Main function to run all visualization methods."""
    print("🎨 LANGGRAPH VISUALIZATION TOOL")
    print("="*50)
    
    # Try the main LangGraph visualization
    visualize_with_langgraph()
    
    print("\n" + "="*50)
    print("🔬 IPYTHON DISPLAY (Jupyter only):")
    print("="*50)
    
    # Try IPython display (will fail outside Jupyter)
    visualize_with_ipython()
    
    print("\n" + "="*50)
    print("💡 TIPS:")
    print("="*50)
    print("• For PNG visualization, you may need: pip install pygraphviz")
    print("• For Jupyter display, run this in a Jupyter notebook")
    print("• Check the generated .mmd file for Mermaid diagram text")
    print("• Use the ASCII representation as a fallback")

if __name__ == "__main__":
    main()
