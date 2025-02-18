# feature_importance_plot.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, feature_columns):
    """
    Create an enhanced bar plot of feature importances from the trained model
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame of features and their importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
    
    # Add importance category
    def get_importance_category(value):
        if value >= 0.12:
            return 'High'
        elif value >= 0.06:
            return 'Medium'
        else:
            return 'Low'
    
    feature_importance_df['Category'] = feature_importance_df['Importance'].apply(get_importance_category)
    
    # Print feature importances with categories
    print("\nFeature Importance Rankings:")
    print("=" * 50)
    for category in ['High', 'Medium', 'Low']:
        print(f"\n{category} Importance Features:")
        print("-" * 30)
        category_df = feature_importance_df[feature_importance_df['Category'] == category]
        print(category_df[['Feature', 'Importance']].to_string(index=False))
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    # Define colors for each category
    colors = {
        'High': '#2ecc71',    # Green
        'Medium': '#3498db',  # Blue
        'Low': '#95a5a6'      # Gray
    }
    
    # Create bar plot with color coding
    bars = plt.barh(feature_importance_df['Feature'], 
                   feature_importance_df['Importance'],
                   color=[colors[cat] for cat in feature_importance_df['Category']])
    
    # Add value labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                f'{width:.1%}',  # Format as percentage
                ha='left', va='center',
                fontweight='bold', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))
    
    # Customize the plot
    plt.title('Feature Importance in Credit Risk Prediction\nGrouped by Importance Level', 
             fontsize=14, pad=20, fontweight='bold')
    plt.xlabel('Importance Score (Percentage)', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    
    # Add a legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[cat], label=f'{cat} Importance')
                      for cat in ['High', 'Medium', 'Low']]
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Adjust layout and grid
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Add vertical lines for importance categories
    plt.axvline(x=0.12, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0.06, color='gray', linestyle='--', alpha=0.5)
    
    # Save plot with high DPI for better quality
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    return feature_importance_df