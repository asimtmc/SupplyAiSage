import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from datetime import datetime, timedelta
from utils.planning import run_what_if_scenario, calculate_material_requirements, generate_production_plan
from utils.visualization import plot_what_if_comparison, plot_material_requirements, plot_production_plan

# Set page config
st.set_page_config(
    page_title="What-If Scenarios",
    page_icon="🎮",
    layout="wide"
)

# Check if data is loaded in session state
if 'sales_data' not in st.session_state or st.session_state.sales_data is None:
    st.warning("Please upload sales data on the main page first.")
    st.stop()

if 'forecasts' not in st.session_state or not st.session_state.forecasts:
    st.warning("Please run forecast analysis on the Demand Forecasting page first.")
    st.stop()

if 'bom_data' not in st.session_state or st.session_state.bom_data is None:
    st.warning("Please upload Bill of Materials (BOM) data on the main page first.")
    st.stop()

if 'supplier_data' not in st.session_state or st.session_state.supplier_data is None:
    st.warning("Please upload Supplier data on the main page first.")
    st.stop()

# Page title
st.title("What-If Scenario Simulator")
st.markdown("""
This simulator allows you to test different supply chain scenarios and see the impact on your operations.
Simulate supplier delays, demand spikes, or material shortages to prepare contingency plans.
""")

# Initialize session state variables for this page
if 'base_scenario' not in st.session_state:
    st.session_state.base_scenario = None
if 'what_if_scenario' not in st.session_state:
    st.session_state.what_if_scenario = None

# Create sidebar for scenario settings
with st.sidebar:
    st.header("Scenario Setup")
    
    # Scenario selection
    scenario_type = st.selectbox(
        "Select Scenario Type",
        options=[
            "Supplier Delay",
            "Demand Increase",
            "Material Shortage"
        ]
    )
    
    # Parameters specific to each scenario type
    if scenario_type == "Supplier Delay":
        st.subheader("Supplier Delay Parameters")
        
        # Get list of suppliers if available
        if 'supplier_data' in st.session_state and st.session_state.supplier_data is not None and 'supplier_id' in st.session_state.supplier_data.columns:
            supplier_list = ['All Suppliers'] + list(st.session_state.supplier_data['supplier_id'].unique())
            selected_supplier = st.selectbox("Select Supplier", options=supplier_list)
            
            if selected_supplier == 'All Suppliers':
                supplier_id = None
            else:
                supplier_id = selected_supplier
        else:
            supplier_id = None
            st.info("No supplier data available. Will simulate delay for all suppliers.")
        
        delay_days = st.slider(
            "Delay (Days)",
            min_value=1,
            max_value=60,
            value=14,
            step=1
        )
        
        # Create scenario config
        scenario_config = {
            'type': 'supplier_delay',
            'supplier_id': supplier_id,
            'delay_days': delay_days
        }
    
    elif scenario_type == "Demand Increase":
        st.subheader("Demand Increase Parameters")
        
        # Get list of SKUs
        sku_list = ['All SKUs'] + list(st.session_state.forecasts.keys())
        selected_sku = st.selectbox("Select SKU", options=sku_list)
        
        if selected_sku == 'All SKUs':
            sku = None
        else:
            sku = selected_sku
        
        increase_percent = st.slider(
            "Demand Increase (%)",
            min_value=5,
            max_value=200,
            value=30,
            step=5
        )
        
        # Create scenario config
        scenario_config = {
            'type': 'demand_increase',
            'sku': sku,
            'increase_percent': increase_percent
        }
    
    elif scenario_type == "Material Shortage":
        st.subheader("Material Shortage Parameters")
        
        # Get list of materials
        if 'bom_data' in st.session_state and st.session_state.bom_data is not None:
            material_list = list(st.session_state.bom_data['material_id'].unique())
            selected_material = st.selectbox("Select Material", options=material_list)
            
            material_id = selected_material
        else:
            material_id = None
            st.error("No BOM data available for material selection.")
        
        reduction_percent = st.slider(
            "Availability Reduction (%)",
            min_value=10,
            max_value=100,
            value=50,
            step=10
        )
        
        # Create scenario config
        scenario_config = {
            'type': 'material_shortage',
            'material_id': material_id,
            'reduction_percent': reduction_percent
        }
    
    # Run scenario button
    if st.button("Run Scenario Analysis"):
        with st.spinner("Simulating scenario..."):
            # Generate base scenario first if not already done
            if st.session_state.base_scenario is None:
                # Generate production plan and material requirements for base scenario
                base_production_plan = generate_production_plan(
                    st.session_state.forecasts,
                    st.session_state.bom_data,
                    st.session_state.supplier_data
                )
                
                base_materials = calculate_material_requirements(
                    st.session_state.forecasts,
                    st.session_state.bom_data,
                    st.session_state.supplier_data
                )
                
                st.session_state.base_scenario = {
                    'production_plan': base_production_plan,
                    'material_requirements': base_materials,
                    'scenario': {'type': 'base'}
                }
            
            # Run what-if scenario
            st.session_state.what_if_scenario = run_what_if_scenario(
                st.session_state.forecasts,
                st.session_state.bom_data,
                st.session_state.supplier_data,
                scenario_config
            )
            
            st.success("Scenario analysis completed!")

# Main content
if st.session_state.base_scenario is not None and st.session_state.what_if_scenario is not None:
    # Scenario comparison section
    st.header("Scenario Comparison")
    
    # Show scenario parameters
    what_if_type = st.session_state.what_if_scenario['scenario'].get('type', 'unknown')
    
    if what_if_type == 'supplier_delay':
        supplier_id = st.session_state.what_if_scenario['scenario'].get('supplier_id', 'All Suppliers')
        delay_days = st.session_state.what_if_scenario['scenario'].get('delay_days', 0)
        scenario_desc = f"Supplier Delay: {'All Suppliers' if supplier_id is None else supplier_id} delayed by {delay_days} days"
    
    elif what_if_type == 'demand_increase':
        sku = st.session_state.what_if_scenario['scenario'].get('sku', 'All SKUs')
        increase_percent = st.session_state.what_if_scenario['scenario'].get('increase_percent', 0)
        scenario_desc = f"Demand Increase: {'All SKUs' if sku is None else sku} increased by {increase_percent}%"
    
    elif what_if_type == 'material_shortage':
        material_id = st.session_state.what_if_scenario['scenario'].get('material_id', 'Unknown')
        reduction_percent = st.session_state.what_if_scenario['scenario'].get('reduction_percent', 0)
        scenario_desc = f"Material Shortage: {material_id} reduced by {reduction_percent}%"
    
    else:
        scenario_desc = "Unknown Scenario"
    
    st.subheader(f"Scenario: {scenario_desc}")
    
    # Production plan comparison
    st.subheader("Production Plan Impact")
    
    # Create comparison chart
    comparison_fig = plot_what_if_comparison(
        st.session_state.base_scenario,
        st.session_state.what_if_scenario
    )
    st.plotly_chart(comparison_fig, use_container_width=True)
    
    # Impact analysis metrics
    st.subheader("Impact Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    # Calculate metrics for comparison
    base_prod = st.session_state.base_scenario['production_plan']
    what_if_prod = st.session_state.what_if_scenario['production_plan']
    
    base_total = base_prod['production_quantity'].sum()
    what_if_total = what_if_prod['production_quantity'].sum()
    
    production_change = what_if_total - base_total
    production_change_pct = (production_change / base_total * 100) if base_total > 0 else 0
    
    # Calculate material requirements changes
    base_mat = st.session_state.base_scenario['material_requirements']
    what_if_mat = st.session_state.what_if_scenario['material_requirements']
    
    base_mat_total = base_mat['order_quantity'].sum() if 'order_quantity' in base_mat.columns else 0
    what_if_mat_total = what_if_mat['order_quantity'].sum() if 'order_quantity' in what_if_mat.columns else 0
    
    material_change = what_if_mat_total - base_mat_total
    material_change_pct = (material_change / base_mat_total * 100) if base_mat_total > 0 else 0
    
    # Display metrics
    with col1:
        st.metric(
            label="Production Volume Impact",
            value=f"{int(what_if_total):,} units",
            delta=f"{production_change_pct:.1f}%",
            delta_color="normal" if production_change_pct >= 0 else "inverse"
        )
    
    with col2:
        st.metric(
            label="Material Requirements Impact",
            value=f"{int(what_if_mat_total):,} units",
            delta=f"{material_change_pct:.1f}%",
            delta_color="normal" if material_change_pct >= 0 else "inverse"
        )
    
    with col3:
        # Calculate service level impact (approximation based on production fulfillment)
        if what_if_type == 'demand_increase':
            # For demand increase, increased production is good
            service_impact = "Positive" if production_change_pct > 0 else "Negative"
            service_value = "Maintained" if production_change_pct >= 0 else "At Risk"
            delta_color = "normal" if production_change_pct >= 0 else "inverse"
        elif what_if_type == 'supplier_delay' or what_if_type == 'material_shortage':
            # For delays and shortages, service level might be at risk
            service_impact = "At Risk" if production_change_pct < 0 else "Maintained"
            service_value = f"{max(0, 100 + production_change_pct):.1f}%" if production_change_pct < 0 else "100%"
            delta_color = "inverse" if production_change_pct < 0 else "normal"
        else:
            service_impact = "Unknown"
            service_value = "N/A"
            delta_color = "off"
        
        st.metric(
            label="Service Level Impact",
            value=service_value,
            delta=service_impact,
            delta_color=delta_color
        )
    
    # Create tabs for detailed views
    tab1, tab2 = st.tabs(["Production Changes", "Material Requirement Changes"])
    
    with tab1:
        # Compare production plans in detail
        st.subheader("Detailed Production Changes")
        
        # Merge base and what-if production plans
        if len(base_prod) > 0 and len(what_if_prod) > 0:
            # Prepare DataFrames for merging
            base_prod_compare = base_prod[['sku', 'period', 'production_quantity']].copy()
            base_prod_compare.rename(columns={'production_quantity': 'base_quantity'}, inplace=True)
            
            what_if_prod_compare = what_if_prod[['sku', 'period', 'production_quantity']].copy()
            what_if_prod_compare.rename(columns={'production_quantity': 'what_if_quantity'}, inplace=True)
            
            # Merge on SKU and period
            prod_comparison = pd.merge(
                base_prod_compare,
                what_if_prod_compare,
                on=['sku', 'period'],
                how='outer'
            ).fillna(0)
            
            # Calculate difference and percentage change
            prod_comparison['difference'] = prod_comparison['what_if_quantity'] - prod_comparison['base_quantity']
            prod_comparison['percent_change'] = (prod_comparison['difference'] / prod_comparison['base_quantity'] * 100).fillna(0)
            
            # Sort by absolute change
            prod_comparison = prod_comparison.sort_values('difference', key=abs, ascending=False)
            
            # Display table
            st.dataframe(prod_comparison, use_container_width=True)
            
            # Show chart of most impacted SKUs
            top_impacted = prod_comparison.head(10)
            
            fig = px.bar(
                top_impacted,
                x='sku',
                y='percent_change',
                title='Top 10 Most Impacted SKUs (% Change)',
                color='percent_change',
                color_continuous_scale=px.colors.diverging.RdBu,
                color_continuous_midpoint=0
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No production data available for comparison")
    
    with tab2:
        # Compare material requirements in detail
        st.subheader("Detailed Material Requirement Changes")
        
        # Merge base and what-if material requirements
        if len(base_mat) > 0 and len(what_if_mat) > 0 and 'order_quantity' in base_mat.columns and 'order_quantity' in what_if_mat.columns:
            # Prepare DataFrames for merging
            base_mat_compare = base_mat[['material_id', 'order_quantity']].copy()
            base_mat_compare = base_mat_compare.groupby('material_id')['order_quantity'].sum().reset_index()
            base_mat_compare.rename(columns={'order_quantity': 'base_quantity'}, inplace=True)
            
            what_if_mat_compare = what_if_mat[['material_id', 'order_quantity']].copy()
            what_if_mat_compare = what_if_mat_compare.groupby('material_id')['order_quantity'].sum().reset_index()
            what_if_mat_compare.rename(columns={'order_quantity': 'what_if_quantity'}, inplace=True)
            
            # Merge on material_id
            mat_comparison = pd.merge(
                base_mat_compare,
                what_if_mat_compare,
                on='material_id',
                how='outer'
            ).fillna(0)
            
            # Calculate difference and percentage change
            mat_comparison['difference'] = mat_comparison['what_if_quantity'] - mat_comparison['base_quantity']
            mat_comparison['percent_change'] = (mat_comparison['difference'] / mat_comparison['base_quantity'] * 100).fillna(0)
            
            # Sort by absolute change
            mat_comparison = mat_comparison.sort_values('difference', key=abs, ascending=False)
            
            # Display table
            st.dataframe(mat_comparison, use_container_width=True)
            
            # Show chart of most impacted materials
            top_materials = mat_comparison.head(10)
            
            fig = px.bar(
                top_materials,
                x='material_id',
                y='percent_change',
                title='Top 10 Most Impacted Materials (% Change)',
                color='percent_change',
                color_continuous_scale=px.colors.diverging.RdBu,
                color_continuous_midpoint=0
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No material requirements data available for comparison")
    
    # Recommended actions section
    st.header("Recommended Actions")
    
    # Generate recommendations based on scenario type and impact
    recommendations = []
    
    if what_if_type == 'supplier_delay':
        recommendations.append("Identify alternative suppliers for critical materials")
        recommendations.append("Increase safety stock levels for materials with long lead times")
        recommendations.append("Adjust production schedule to prioritize products with available materials")
        
        if production_change_pct < -10:
            recommendations.append("Alert customers about potential delivery delays")
            recommendations.append("Consider expedited shipping for critical materials")
    
    elif what_if_type == 'demand_increase':
        if production_change_pct < 0:
            recommendations.append("Evaluate capacity constraints and consider overtime or additional shifts")
            recommendations.append("Prioritize high-margin or strategic customer orders")
            recommendations.append("Accelerate material orders to meet increased demand")
        else:
            recommendations.append("Monitor inventory levels to ensure sufficient materials")
            recommendations.append("Verify production capacity can handle the increased load")
            recommendations.append("Update forecasts to reflect the new demand pattern")
    
    elif what_if_type == 'material_shortage':
        recommendations.append("Identify alternative materials or suppliers")
        recommendations.append("Reformulate products to reduce dependency on constrained materials")
        recommendations.append("Adjust production schedule to prioritize products not affected by the shortage")
        
        if production_change_pct < -15:
            recommendations.append("Communicate with customers about potential impacts")
            recommendations.append("Consider price adjustments to manage demand during the shortage")
    
    # Display recommendations
    for i, recommendation in enumerate(recommendations):
        st.markdown(f"**{i+1}. {recommendation}**")
    
    # Export scenario results
    st.header("Export Scenario Analysis")
    
    if st.button("Prepare Scenario Report"):
        # Create a summary report
        summary_data = {
            'Metric': [
                'Scenario Type',
                'Scenario Parameters',
                'Base Production Volume',
                'Scenario Production Volume',
                'Production Volume Change',
                'Production Volume Change (%)',
                'Base Material Requirements',
                'Scenario Material Requirements',
                'Material Requirements Change',
                'Material Requirements Change (%)'
            ],
            'Value': [
                what_if_type.replace('_', ' ').title(),
                scenario_desc,
                f"{int(base_total):,} units",
                f"{int(what_if_total):,} units",
                f"{int(production_change):,} units",
                f"{production_change_pct:.1f}%",
                f"{int(base_mat_total):,} units",
                f"{int(what_if_mat_total):,} units",
                f"{int(material_change):,} units",
                f"{material_change_pct:.1f}%"
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Display preview
        st.subheader("Scenario Summary")
        st.dataframe(summary_df, use_container_width=True)
        
        # Add recommendations to report
        recommendations_data = [{'Recommendation': rec} for rec in recommendations]
        recommendations_df = pd.DataFrame(recommendations_data)
        
        st.subheader("Recommended Actions")
        st.dataframe(recommendations_df, use_container_width=True)
        
        # Convert to Excel for download
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            summary_df.to_excel(writer, sheet_name='Scenario Summary', index=False)
            recommendations_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            if len(prod_comparison) > 0:
                prod_comparison.to_excel(writer, sheet_name='Production Changes', index=False)
            
            if len(mat_comparison) > 0:
                mat_comparison.to_excel(writer, sheet_name='Material Changes', index=False)
        
        excel_buffer.seek(0)
        
        # Create download button
        st.download_button(
            label="Download Scenario Report as Excel",
            data=excel_buffer,
            file_name=f"scenario_analysis_{what_if_type}_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.ms-excel"
        )

else:
    # Show instructions when no scenario has been run
    st.info("👈 Please configure and run a scenario analysis using the sidebar.")
    
    # Show example what-if scenarios
    st.header("Example Scenarios to Consider")
    
    st.markdown("""
    ### 1. Supplier Delay
    Simulate the impact of supplier shipping delays on your production schedule and inventory levels.
    
    **Example Use Cases:**
    - A key supplier experiencing manufacturing issues
    - Transportation delays due to weather events or port congestion
    - Geopolitical disruptions affecting international shipping
    
    **Key Impacts to Analyze:**
    - Production disruptions
    - Inventory shortages
    - Service level deterioration
    """)
    
    st.markdown("""
    ### 2. Demand Increase
    Model sudden spikes in customer demand and test your supply chain's ability to scale.
    
    **Example Use Cases:**
    - Marketing promotion generating unexpected sales
    - Competitor exit creating market share opportunity
    - Seasonal demand exceeding forecast
    
    **Key Impacts to Analyze:**
    - Production capacity constraints
    - Material shortages
    - Fulfillment rate changes
    """)
    
    st.markdown("""
    ### 3. Material Shortage
    Evaluate the consequences of raw material constraints on your production capabilities.
    
    **Example Use Cases:**
    - Global shortage of key components
    - Quality issues requiring material rejection
    - Supplier bankruptcy or force majeure event
    
    **Key Impacts to Analyze:**
    - Production replanning needs
    - Customer order prioritization
    - Financial impact of shortages
    """)
    
    st.markdown("""
    Click the "Run Scenario Analysis" button in the sidebar to simulate any of these scenarios using your actual supply chain data.
    """)
