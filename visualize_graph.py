#!/usr/bin/env python3

"""
Graph Visualization Script for Customer Support System
This script imports the graph from main.py and displays it as an image
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(__file__))

def visualize_graph():
    """Import the graph from main.py and visualize it"""
    
    try:
        # Import the graph from main.py
        print("Importing graph from main.py...")
        from main import graph
        
        print("✅ Successfully imported graph")
        print(f"Graph type: {type(graph)}")
        
        # Try to display the graph using IPython
        try:
            from IPython.display import Image, display
            
            print("\nGenerating graph visualization...")
            
            # Generate the Mermaid diagram
            mermaid_diagram = graph.get_graph(xray=True).draw_mermaid_png()
            
            # Display the image
            display(Image(mermaid_diagram))
            
            print("✅ Graph visualization displayed successfully!")
            
        except ImportError:
            print("❌ IPython not available. Trying alternative visualization methods...")
            
            # Alternative: Save the graph as a file
            try:
                # Save as Mermaid text
                mermaid_text = graph.get_graph(xray=True).draw_mermaid()
                with open("graph_diagram.mmd", "w") as f:
                    f.write(mermaid_text)
                print("✅ Mermaid diagram saved to 'graph_diagram.mmd'")
                print("You can view it at: https://mermaid.live/")
                
                # Also try to save as PNG if possible
                try:
                    png_data = graph.get_graph(xray=True).draw_mermaid_png()
                    with open("graph_diagram.png", "wb") as f:
                        f.write(png_data)
                    print("✅ PNG diagram saved to 'graph_diagram.png'")
                except Exception as e:
                    print(f"⚠️  Could not save PNG: {e}")
                    
            except Exception as e:
                print(f"❌ Could not save diagram: {e}")
                
        except Exception as e:
            print(f"❌ Error displaying graph: {e}")
            print("Trying to save diagram instead...")
            
            try:
                # Save as Mermaid text
                mermaid_text = graph.get_graph(xray=True).draw_mermaid()
                with open("graph_diagram.mmd", "w") as f:
                    f.write(mermaid_text)
                print("✅ Mermaid diagram saved to 'graph_diagram.mmd'")
                print("You can view it at: https://mermaid.live/")
            except Exception as save_e:
                print(f"❌ Could not save diagram: {save_e}")
        
        # Print graph structure information
        print("\n📊 Graph Structure Information:")
        print(f"Number of nodes: {len(graph.nodes)}")
        print(f"Number of edges: {len(graph.edges)}")
        print(f"Start node: {graph.start}")
        print(f"End node: {graph.end}")
        
        print("\n🔗 Nodes:")
        for node_name in graph.nodes:
            print(f"  - {node_name}")
            
        print("\n🔄 Edges:")
        for edge in graph.edges:
            print(f"  - {edge}")
            
    except ImportError as e:
        print(f"❌ Failed to import from main.py: {e}")
        print("Make sure you're running this script from the same directory as main.py")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

def save_graph_files():
    """Save the graph in multiple formats for offline viewing"""
    
    try:
        from main import graph
        
        print("\n💾 Saving graph files...")
        
        # Save as Mermaid text
        mermaid_text = graph.get_graph(xray=True).draw_mermaid()
        with open("graph_diagram.mmd", "w") as f:
            f.write(mermaid_text)
        print("✅ Mermaid diagram saved to 'graph_diagram.mmd'")
        
        # Save as PNG
        try:
            png_data = graph.get_graph(xray=True).draw_mermaid_png()
            with open("graph_diagram.png", "wb") as f:
                f.write(png_data)
            print("✅ PNG diagram saved to 'graph_diagram.png'")
        except Exception as e:
            print(f"⚠️  Could not save PNG: {e}")
        
        # Save as DOT format (Graphviz)
        try:
            dot_text = graph.get_graph(xray=True).draw_dot()
            with open("graph_diagram.dot", "w") as f:
                f.write(dot_text)
            print("✅ DOT diagram saved to 'graph_diagram.dot'")
        except Exception as e:
            print(f"⚠️  Could not save DOT: {e}")
            
        print("\n📁 Files saved successfully!")
        print("You can view the Mermaid diagram at: https://mermaid.live/")
        print("You can view the DOT diagram at: https://dreampuf.github.io/GraphvizOnline/")
        
    except Exception as e:
        print(f"❌ Error saving graph files: {e}")

if __name__ == "__main__":
    print("🚀 Customer Support System Graph Visualizer")
    print("=" * 50)
    
    # Try to visualize the graph
    visualize_graph()
    
    # Also save the graph files
    save_graph_files()
    
    print("\n✨ Visualization complete!")

