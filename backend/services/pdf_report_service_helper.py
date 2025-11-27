
def _create_venture_bar_chart(m3_eval: Dict) -> List:
    """Create bar chart for venture categories"""
    elements = []
    
    cat_scores = m3_eval.get("category_scores", {})
    if not cat_scores:
        return []
        
    # Create bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    
    categories = list(cat_scores.keys())
    scores = list(cat_scores.values())
    
    # Color bars by score
    bar_colors = ['#2ECC71' if s >= 80 else '#F39C12' if s >= 60 else '#E74C3C' for s in scores]
    
    bars = ax.barh(categories, scores, color=bar_colors, alpha=0.8)
    ax.set_xlabel('Score', fontsize=10)
    ax.set_xlim(0, 100)
    ax.axvline(x=60, color='red', linestyle='--', alpha=0.3, linewidth=1)
    ax.axvline(x=80, color='green', linestyle='--', alpha=0.3, linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score + 2, i, f'{score:.1f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save to buffer
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    img_buffer.seek(0)
    
    # Add to PDF
    img = Image(img_buffer, width=5*inch, height=2.5*inch)
    elements.append(img)
    
    return elements
