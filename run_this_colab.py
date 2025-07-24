  #!/usr/bin/env python3
  """
  Clean HAAM Analysis Script (v1.2.0)
  ===================================
  Run complete HAAM analysis with comprehensive metrics and enhanced 
  visualizations.
  """

  # Mount Google Drive
  from google.colab import drive
  drive.mount('/content/drive')

  import pandas as pd
  import numpy as np
  import os
  import matplotlib.pyplot as plt
  import seaborn as sns
  from haam import HAAM

  # Set plotting style
  plt.style.use('seaborn-v0_8-whitegrid')
  plt.rcParams.update({
      'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
      'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 9
  })

  print("="*80)
  print("HAAM ANALYSIS v1.2.0 - WITH COMPREHENSIVE METRICS & ENHANCED 
  VISUALIZATIONS")
  print("="*80)

  # =======================================================================
  =======
  # LOAD DATA
  # =======================================================================
  =======

  print("\n1. LOADING DATA...")
  print("-"*60)

  # Load the complete dataset with text
  filename = 'essay_embeddings_minilm_with_text.csv'
  data_path = f'/content/drive/MyDrive/2025_06_30_anonymized_data_dmllme_20
  25_06_30/{filename}'

  if not os.path.exists(data_path):
      raise FileNotFoundError(f"Data file not found: {data_path}")

  df = pd.read_csv(data_path)
  print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

  # Extract data
  texts = df['text'].values.tolist()
  embedding_cols = [f'embed_dim_{i}' for i in range(384)]
  embeddings = df[embedding_cols].values
  social_class = df['social_class'].astype(float).values
  ai_rating = df['ai_rating'].astype(float).values
  human_rating = df['human_rating'].astype(float).values

  print(f"\nData summary:")
  print(f"  Texts: {len(texts)} essays")
  print(f"  Embeddings: {embeddings.shape}")
  print(f"  Valid criterion (Y): {(~np.isnan(social_class)).sum()}")
  print(f"  Valid AI ratings: {(~np.isnan(ai_rating)).sum()}")
  print(f"  Valid human ratings: {(~np.isnan(human_rating)).sum()}")

  # Create output directory
  output_dir = 'haam_results'
  os.makedirs(output_dir, exist_ok=True)

  # =======================================================================
  =======
  # RUN HAAM ANALYSIS
  # =======================================================================
  =======

  print("\n2. RUNNING HAAM ANALYSIS...")
  print("-"*60)

  # Initialize and run HAAM with new default parameters
  # Note: New defaults match my_colab.py exactly
  haam = HAAM(
      criterion=social_class,
      ai_judgment=ai_rating,
      human_judgment=human_rating,
      embeddings=embeddings,
      texts=texts,  # Include texts for topic analysis
      n_components=200,
      # New defaults: min_cluster_size=10, min_samples=2
      # UMAP: n_neighbors=5, min_dist=0.0, metric='cosine'
      auto_run=True
  )

  results = haam.results
  print("\n✓ Analysis complete with comprehensive metrics!")

  # =======================================================================
  =======
  # DISPLAY NEW COMPREHENSIVE METRICS
  # =======================================================================
  =======

  print("\n3. COMPREHENSIVE METRICS...")
  print("-"*60)

  # Display Total Effects (DML coefficients)
  if 'total_effects' in haam.analysis.results:
      print("\n📊 TOTAL EFFECTS (DML Coefficients):")
      te = haam.analysis.results['total_effects']
      if 'Y_AI' in te:
          print(f"  Y → AI: β = {te['Y_AI']['coefficient']:.3f} (SE = {te['Y_AI']['se']:.3f})")
          print(f"         β_check = {te['Y_AI']['check_beta']:.3f}")
      if 'Y_HU' in te:
          print(f"  Y → HU: β = {te['Y_HU']['coefficient']:.3f} (SE = {te['Y_HU']['se']:.3f})")
          print(f"         β_check = {te['Y_HU']['check_beta']:.3f}")
      if 'HU_AI' in te:
          print(f"  HU → AI: β = {te['HU_AI']['coefficient']:.3f} (SE = {te['HU_AI']['se']:.3f})")
          print(f"          β_check = {te['HU_AI']['check_beta']:.3f}")

  # Display Residual Correlations
  if 'residual_correlations' in haam.analysis.results:
      print("\n📊 RESIDUAL CORRELATIONS (C's):")
      rc = haam.analysis.results['residual_correlations']
      print(f"  C(AI,HU) = {rc.get('AI_HU', 0):.3f}  [corr(e_AI, e_HU) 
  after controlling for Y]")
      print(f"  C(Y,AI) = {rc.get('Y_AI', 0):.3f}   [corr(e_Y, e_AI) after 
  controlling for HU]")
      print(f"  C(Y,HU) = {rc.get('Y_HU', 0):.3f}   [corr(e_Y, e_HU) after 
  controlling for AI]")

  # Display Policy Similarities
  if 'policy_similarities' in haam.analysis.results:
      print("\n📊 POLICY SIMILARITIES (Prediction Correlations):")
      ps = haam.analysis.results['policy_similarities']
      print(f"  r(Ŷ, ÂI) = {ps.get('Y_AI', 0):.3f}")
      print(f"  r(Ŷ, ĤU) = {ps.get('Y_HU', 0):.3f}")
      print(f"  r(ÂI, ĤU) = {ps.get('AI_HU', 0):.3f}")

  # Display Mediation Analysis Results with Visualization
  print("\n📊 MEDIATION ANALYSIS (PoMA):")
  haam.analysis.display_mediation_results()

  # =======================================================================
  =======
  # SAVE AND DISPLAY RESULTS
  # =======================================================================
  =======

  print("\n4. SAVING RESULTS...")
  print("-"*60)

  # Save coefficients (now uses Y instead of SC)
  if 'coefficients' in results:
      coef_df = results['coefficients']
      coef_path = os.path.join(output_dir, 'coefficients.csv')
      coef_df.to_csv(coef_path, index=False)
      print(f"✓ Saved coefficients: {coef_path}")

      # Display sample
      print("\nSample coefficients (first 5 PCs):")
      print(coef_df.head().to_string())

  # Save model summary
  if 'model_summary' in results:
      summary_df = results['model_summary']
      summary_path = os.path.join(output_dir, 'model_summary.csv')
      summary_df.to_csv(summary_path, index=False)
      print(f"\n✓ Saved model summary: {summary_path}")

      print("\nModel Performance:")
      print(summary_df.to_string(index=False))

  # Save comprehensive metrics summary (NEW in v1.2.0)
  print("\nSaving comprehensive metrics...")
  metrics_dict = haam.create_metrics_summary(output_dir=output_dir)
  print(f"✓ Saved metrics summary: {os.path.join(output_dir, 
  'haam_metrics_summary.json')}")

  # Save topic summaries for ALL 200 PCs (NEW: top 30 / bottom 30)
  if hasattr(haam, 'topic_summaries') and haam.topic_summaries:
      topic_path = os.path.join(output_dir, 'topic_summaries_all_pcs.txt')
      with open(topic_path, 'w') as f:
          f.write("COMPREHENSIVE TOPIC SUMMARIES FOR ALL 200 PRINCIPAL 
  COMPONENTS\n")
          f.write("="*80 + "\n")
          f.write("Showing top 30 and bottom 30 topics for each PC\n")
          f.write("="*80 + "\n\n")

          # Write summaries for ALL PCs
          for pc_idx in sorted(haam.topic_summaries.keys()):
              topics = haam.topic_summaries[pc_idx]
              f.write(f"\nPC{pc_idx + 1}:\n")
              f.write("="*60 + "\n")

              # High topics
              if 'high_topics' in topics and topics['high_topics']:
                  high_topics = [t for t in topics['high_topics'] if t !=
  'No significant high topics']
                  if high_topics:
                      f.write(f"\nHIGH on PC{pc_idx + 1} (Top 
  {len(high_topics)} topics):\n")
                      f.write("-"*40 + "\n")
                      for i, topic in enumerate(high_topics, 1):
                          f.write(f"{i:3d}. {topic}\n")
                  else:
                      f.write("\nHIGH: No significant high topics\n")

              # Low topics
              if 'low_topics' in topics and topics['low_topics']:
                  low_topics = [t for t in topics['low_topics'] if t != 'No
   significant low topics']
                  if low_topics:
                      f.write(f"\nLOW on PC{pc_idx + 1} (Bottom 
  {len(low_topics)} topics):\n")
                      f.write("-"*40 + "\n")
                      for i, topic in enumerate(low_topics, 1):
                          f.write(f"{i:3d}. {topic}\n")
                  else:
                      f.write("\nLOW: No significant low topics\n")

              f.write("\n")

      print(f"\n✓ Saved comprehensive topic summaries: {topic_path}")

  # =======================================================================
  =======
  # CREATE VISUALIZATIONS
  # =======================================================================
  =======

  print("\n5. CREATING VISUALIZATIONS...")
  print("-"*60)

  # Create main visualization with different ranking methods
  ranking_methods = {
      'HU': 'Human judgment ranking (default)',
      'Y': 'Criterion ranking',
      'AI': 'AI judgment ranking',
      'triple': 'Triple method (top 3 from each)'
  }

  for ranking_method, description in ranking_methods.items():
      try:
          print(f"\nCreating visualization with {description}...")

          # Get top PCs using this ranking method
          top_pcs = haam.analysis.get_top_pcs(n_top=9,
  ranking_method=ranking_method)

          # Create visualization
          output_file = os.path.join(output_dir,
  f'haam_main_visualization_{ranking_method.lower()}.html')
          haam.visualizer.create_main_visualization(top_pcs, output_file)

          print(f"✓ Saved: {output_file}")

          # Only create the default one using the main method
          if ranking_method == 'HU':
              main_viz =
  haam.create_main_visualization(output_dir=output_dir)
              print(f"✓ Also saved default: {main_viz}")

      except Exception as e:
          print(f"⚠️  Visualization error for {ranking_method}: 
  {str(e)[:100]}...")

  # Create mini coefficient grid
  try:
      print("\nCreating mini coefficient grid...")
      mini_grid = haam.create_mini_grid(output_dir=output_dir)
      print(f"✓ Saved: {mini_grid}")
  except Exception as e:
      print(f"⚠️  Mini grid error: {str(e)[:100]}...")

  # Create UMAP visualization
  try:
      print("\nCreating UMAP visualizations...")
      for color_by in ['SC', 'AI', 'HU']:
          umap_viz = haam.create_umap_visualization(
              n_components=3,
              color_by=color_by,
              output_dir=output_dir
          )
          print(f"✓ Saved: {umap_viz}")
  except Exception as e:
      print(f"⚠️  UMAP error: {str(e)[:100]}...")

  # Create PC effects visualization
  try:
      print("\nCreating PC effects visualization...")
      pc_effects =
  haam.create_pc_effects_visualization(output_dir=output_dir, n_top=20)
      print(f"✓ Saved: {pc_effects}")
  except Exception as e:
      print(f"⚠️  PC effects error: {str(e)[:100]}...")

  # =======================================================================
  =======
  # OPTIONAL: RE-CREATE VISUALIZATION WITH MANUAL PC NAMES
  # =======================================================================
  =======

  print("\n6. EXAMPLE: RE-CREATE VISUALIZATION WITH INTERPRETIVE NAMES...")
  print("-"*60)

  # After analyzing the results, you can add meaningful names
  # Example based on your top PCs (adjust indices based on your ranking 
  method)
  pc_names_example = {
      # Map PC indices (0-based) to descriptive names
      # Review topic_summaries_all_pcs.txt to determine appropriate names
      # e.g., if PC5 is about lifestyle: 4: "Lifestyle & Work"
  }

  if pc_names_example:
      print("Re-creating main visualization with custom PC names...")
      top_pcs_human = haam.analysis.get_top_pcs(n_top=9,
  ranking_method='HU')
      named_viz_path = os.path.join(output_dir,
  'haam_main_visualization_named.html')
      haam.visualizer.create_main_visualization(
          pc_indices=top_pcs_human,
          output_file=named_viz_path,
          pc_names=pc_names_example
      )
      print(f"✓ Saved named version: {named_viz_path}")
  else:
      print("ℹ️  No custom PC names provided. Edit pc_names_example after 
  interpreting results.")

  # =======================================================================
  =======
  # CREATE CUSTOM VISUALIZATIONS
  # =======================================================================
  =======

  print("\n7. CREATING CUSTOM VISUALIZATIONS...")
  print("-"*60)

  # 1. Coefficient Heatmap (Updated to use Y instead of SC)
  if 'coefficients' in results:
      fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

      # Prepare data - note: internal still uses SC, display uses Y
      coef_matrix = np.zeros((3, 50))  # First 50 PCs
      outcome_map = {'Y': 'SC', 'AI': 'AI', 'HU': 'HU'}

      for i, (display_outcome, internal_outcome) in
  enumerate(outcome_map.items()):
          coef_col = f'{internal_outcome}_coef'
          if coef_col in coef_df.columns:
              coef_matrix[i, :] = coef_df[coef_col].values[:50]

      # Heatmap
      im = ax1.imshow(coef_matrix, aspect='auto', cmap='RdBu_r',
                      vmin=-np.max(np.abs(coef_matrix)),
  vmax=np.max(np.abs(coef_matrix)))
      ax1.set_yticks([0, 1, 2])
      ax1.set_yticklabels(['Criterion (Y)', 'AI Rating', 'Human Rating'])
      ax1.set_xlabel('Principal Component')
      ax1.set_title('Coefficient Heatmap (First 50 PCs)')
      plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

      # Top PCs bar chart
      if 'top_pcs' in results:
          top_10 = results['top_pcs'][:10]
          avg_abs_coef = []
          for pc in top_10:
              coefs = []
              for internal_outcome in ['SC', 'AI', 'HU']:
                  coef_col = f'{internal_outcome}_coef'
                  if coef_col in coef_df.columns:
                      coefs.append(abs(coef_df.iloc[pc][coef_col]))
              avg_abs_coef.append(np.mean(coefs))

          ax2.bar([f'PC{pc+1}' for pc in top_10], avg_abs_coef,
  color='#9467bd')
          ax2.set_xlabel('Principal Component')
          ax2.set_ylabel('Average |Coefficient|')
          ax2.set_title('Top 10 PCs by Human Judgment Ranking')
          ax2.grid(True, alpha=0.3)

      plt.tight_layout()
      plt.savefig(os.path.join(output_dir, 'coefficient_analysis.png'),
  dpi=300, bbox_inches='tight')
      plt.show()

  # 2. Model Performance Dashboard with New Metrics
  if 'model_summary' in results:
      fig, axes = plt.subplots(2, 3, figsize=(18, 10))

      summary_df = results['model_summary']
      outcomes = ['Criterion (Y)', 'AI Rating', 'Human Rating']
      outcome_codes = ['SC', 'AI', 'HU']  # Internal codes

      # Map internal codes to get data
      r2_cv = []
      r2_in = []
      n_selected = []
      for code in outcome_codes:
          row = summary_df[summary_df['Outcome'] == code]
          if not row.empty:
              r2_cv.append(row['R2_cv'].values[0])
              r2_in.append(row['R2_insample'].values[0])
              n_selected.append(row['N_selected'].values[0])

      # R² comparison
      ax = axes[0, 0]
      x = np.arange(len(outcomes))
      width = 0.35
      ax.bar(x - width/2, r2_cv, width, label='Cross-validated',
  color='#1f77b4')
      ax.bar(x + width/2, r2_in, width, label='In-sample', color='#ff7f0e')
      ax.set_ylabel('R²')
      ax.set_title('Model Performance')
      ax.set_xticks(x)
      ax.set_xticklabels(outcomes, rotation=45)
      ax.legend()
      ax.grid(True, alpha=0.3)

      # Feature selection
      ax = axes[0, 1]
      ax.bar(outcomes, n_selected, color=['#2ca02c', '#d62728', '#9467bd'])
      ax.set_ylabel('Number of Selected PCs')
      ax.set_title('Feature Selection (out of 200)')
      ax.axhline(y=200, color='gray', linestyle='--', alpha=0.5)
      ax.grid(True, alpha=0.3)

      # Total Effects
      ax = axes[0, 2]
      if 'total_effects' in haam.analysis.results:
          te = haam.analysis.results['total_effects']
          effects = []
          effect_names = []
          if 'Y_AI' in te:
              effects.append(te['Y_AI']['coefficient'])
              effect_names.append('Y→AI')
          if 'Y_HU' in te:
              effects.append(te['Y_HU']['coefficient'])
              effect_names.append('Y→HU')
          if 'HU_AI' in te:
              effects.append(te['HU_AI']['coefficient'])
              effect_names.append('HU→AI')

          colors = ['#be123c', '#d97706', '#9333ea']
          ax.bar(effect_names, effects, color=colors[:len(effects)])
          ax.set_ylabel('Total Effect (β)')
          ax.set_title('DML Total Effects')
          ax.grid(True, alpha=0.3)

      # Residual Correlations
      ax = axes[1, 0]
      if 'residual_correlations' in haam.analysis.results:
          rc = haam.analysis.results['residual_correlations']
          corrs = [rc.get('AI_HU', 0), rc.get('Y_AI', 0), rc.get('Y_HU',
  0)]
          corr_names = ['C(AI,HU)', 'C(Y,AI)', 'C(Y,HU)']
          colors = ['#9333ea', '#334155', '#334155']
          ax.bar(corr_names, corrs, color=colors)
          ax.set_ylabel('Residual Correlation')
          ax.set_title('Residual Correlations (C)')
          ax.axhline(y=0, color='black', linewidth=0.5)
          ax.grid(True, alpha=0.3)

      # Policy Similarities
      ax = axes[1, 1]
      if 'policy_similarities' in haam.analysis.results:
          ps = haam.analysis.results['policy_similarities']
          sims = [ps.get('Y_AI', 0), ps.get('Y_HU', 0), ps.get('AI_HU', 0)]
          sim_names = ['r(Ŷ,ÂI)', 'r(Ŷ,ĤU)', 'r(ÂI,ĤU)']
          ax.bar(sim_names, sims, color='#64748b')
          ax.set_ylabel('Correlation')
          ax.set_title('Policy Similarities')
          ax.grid(True, alpha=0.3)

      # Summary text
      ax = axes[1, 2]
      ax.axis('off')
      summary_text = f"SUMMARY\n{'='*30}\n\n"
      summary_text += f"Avg R²(CV): {np.mean(r2_cv):.3f}\n"
      summary_text += f"Avg PCs selected: {np.mean(n_selected):.0f}\n\n"

      if 'total_effects' in haam.analysis.results and 'Y_AI' in
  haam.analysis.results['total_effects']:
          summary_text += f"Total Effect Y→AI: 
  {haam.analysis.results['total_effects']['Y_AI']['coefficient']:.3f}\n"
      if 'residual_correlations' in haam.analysis.results:
          summary_text += f"C(AI,HU): 
  {haam.analysis.results['residual_correlations']['AI_HU']:.3f}\n"

      summary_text += f"\nBest model: {outcomes[np.argmax(r2_cv)]}"
      summary_text += f"\nR²(CV) = {np.max(r2_cv):.3f}"

      ax.text(0.1, 0.9, summary_text, ha='left', va='top', fontsize=11,
              transform=ax.transAxes, family='monospace',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray",
  alpha=0.5))

      plt.suptitle('Enhanced Model Performance Dashboard', fontsize=14,
  fontweight='bold')
      plt.tight_layout()
      plt.savefig(os.path.join(output_dir,
  'performance_dashboard_enhanced.png'), dpi=300, bbox_inches='tight')
      plt.show()

  # =======================================================================
  =======
  # FINAL SUMMARY
  # =======================================================================
  =======

  print("\n" + "="*80)
  print("ANALYSIS COMPLETE!")
  print("="*80)

  print(f"\nAll results saved to: {output_dir}/")
  print("\nGenerated files:")
  print("  📊 coefficients.csv - Full coefficient matrix")
  print("  📈 model_summary.csv - Model performance metrics")
  print("  📝 topic_summaries_all_pcs.txt - Top 30 & bottom 30 topics for 
  ALL 200 PCs")
  print("  📋 haam_metrics_summary.json - Comprehensive metrics 
  including:")
  print("     • Total Effects (DML coefficients)")
  print("     • DML check betas")
  print("     • Residual correlations (C's)")
  print("     • Policy similarities")
  print("     • Mediation analysis (PoMA)")
  print("  🎨 HTML visualizations:")
  print("     • Main visualizations for each ranking method (HU, Y, AI, 
  triple)")
  print("     • Mini grid, UMAP (3 versions), PC effects")
  print("  📊 coefficient_analysis.png - Coefficient heatmap and top PCs")
  print("  📈 performance_dashboard_enhanced.png - Enhanced dashboard with 
  all metrics")

  # Display final key metrics
  print("\n📋 FINAL KEY METRICS:")
  print("-"*60)

  if 'model_performance' in metrics_dict:
      print("\nModel Performance (R² cross-validated):")
      for outcome, perf in metrics_dict['model_performance'].items():
          print(f"  {outcome}: {perf['r2_cv']:.3f}")

  if 'total_effects' in metrics_dict:
      print("\nTotal Effects (DML):")
      for path, effect in metrics_dict['total_effects'].items():
          print(f"  {path}: β = {effect['coefficient']:.3f}, β_check = 
  {effect['check_beta']:.3f}")

  if 'residual_correlations' in metrics_dict:
      print("\nResidual Correlations:")
      for pair, corr in metrics_dict['residual_correlations'].items():
          print(f"  C({pair}): {corr:.3f}")

  if 'mediation_analysis' in metrics_dict and
  metrics_dict['mediation_analysis']:
      print("\nProportion Mediated (PoMA):")
      for outcome, med in metrics_dict['mediation_analysis'].items():
          if 'proportion_mediated' in med:
              print(f"  {outcome}: {med['proportion_mediated']:.1f}%")

  print("\n✅ Analysis complete! Review topic_summaries_all_pcs.txt to 
  interpret PCs.")
  print("💡 Then use pc_names parameter to add meaningful labels to 
  visualization.")
  print("📊 All comprehensive metrics are now included in the 
  visualizations!")
