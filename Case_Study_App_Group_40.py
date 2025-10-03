# IMPORTANT: please check README.md for setup instructions in additional_files/


"""
Automotive Production Lead Time Analysis Dashboard
Case Study IDA Group 40

This application analyzes vehicle production lead times to identify vehicle types 
that exhibit extended production cycles from component manufacturing to final assembly.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table
import numpy as np
import base64
import os
from datetime import datetime
import webbrowser
from threading import Timer

# Application Configuration
APP_TITLE = "Automotive Production Lead Time Analysis"
DATA_FILE = "Final_dataset_group_40.csv"
LOGO_PATH = "www/QW_print.png"

# Corporate Design Configuration
CORPORATE_COLORS = {
    'primary': '#87CEEB',
    'secondary': '#4682B4',
    'accent': '#B0E0E6',
    'dark': '#2F4F4F',
    'light': '#F0F8FF',
    'white': '#FFFFFF',
    'warning': '#FFA500',
    'success': '#32CD32'
}

FONT_FAMILY = ('"Source Sans Pro", -apple-system, BlinkMacSystemFont, '
               '"Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif')

# Initialize Dash Application
app = dash.Dash(__name__,
                title=APP_TITLE,
                update_title=None,
                external_stylesheets=[
                    ('https://fonts.googleapis.com/css2?family=Source+'
                     'Sans+Pro:wght@300;400;600;700&display=swap')
                ])

def load_and_prepare_data():
    """Load and prepare vehicle production data for analysis."""
    try:
        df = pd.read_csv(DATA_FILE)
        
        df['Vehicle_Prod_Date'] = pd.to_datetime(df['Vehicle_Prod_Date'])
        df['Earliest_Part_Date'] = pd.to_datetime(df['Earliest_Part_Date'])
        
        df['Vehicle_Type'] = df['Source_Table'].str.replace('fahrzeuge_', '').str.upper()
        
        # Convert component latest production dates to datetime
        latest_part_cols = ['Karosserie_Latest_Part_Prod_Date', 'Schaltung_Latest_Part_Prod_Date',
                           'Sitze_Latest_Part_Prod_Date', 'Motor_Latest_Part_Prod_Date']
        earliest_part_cols = ['Karosserie_Earliest_Part_Prod_Date', 'Schaltung_Earliest_Part_Prod_Date',
                              'Sitze_Earliest_Part_Prod_Date', 'Motor_Earliest_Part_Prod_Date']
        
        for col in latest_part_cols + earliest_part_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Compute component-specific lead times as (Latest - Earliest) for each component
        df['New_Karosserie_LeadTime'] = (df['Karosserie_Latest_Part_Prod_Date'] - df['Karosserie_Earliest_Part_Prod_Date']).dt.days
        df['New_Schaltung_LeadTime'] = (df['Schaltung_Latest_Part_Prod_Date'] - df['Schaltung_Earliest_Part_Prod_Date']).dt.days
        df['New_Sitze_LeadTime'] = (df['Sitze_Latest_Part_Prod_Date'] - df['Sitze_Earliest_Part_Prod_Date']).dt.days
        df['New_Motor_LeadTime'] = (df['Motor_Latest_Part_Prod_Date'] - df['Motor_Earliest_Part_Prod_Date']).dt.days
        
        # Compute the latest production date across all components for each vehicle
        df['Latest_Part_Prod_Date'] = df[latest_part_cols].max(axis=1)
        
        # Calculate new lead time as Vehicle_Prod_Date - Latest_Part_Prod_Date
        df['New_Lead_Time_Days'] = (df['Vehicle_Prod_Date'] - df['Latest_Part_Prod_Date']).dt.days
        
        df['Production_Year'] = df['Vehicle_Prod_Date'].dt.year
        df['Production_Month'] = df['Vehicle_Prod_Date'].dt.month
        
        # Update lead time categories to use new calculation
        df['Lead_Time_Category'] = pd.cut(df['New_Lead_Time_Days'],
                                          bins=[0, 15, 30, 45, 60, float('inf')],
                                          labels=['Very Fast (≤15d)', 'Fast (16-30d)',
                                                  'Standard (31-45d)', 'Slow (46-60d)',
                                                  'Very Slow (>60d)'])
        
        component_cols = ['New_Karosserie_LeadTime', 'New_Schaltung_LeadTime',
                          'New_Sitze_LeadTime', 'New_Motor_LeadTime']
        df['Max_Component_LeadTime'] = df[component_cols].max(axis=1)
        df['Min_Component_LeadTime'] = df[component_cols].min(axis=1)
        df['Component_LeadTime_Range'] = (df['Max_Component_LeadTime'] -
                                          df['Min_Component_LeadTime'])
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

def encode_logo():
    """Encode logo image for display in the application."""
    try:
        with open(LOGO_PATH, 'rb') as f:
            logo_data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{logo_data}"
    except:
        return None

def create_summary_cards(df):
    """Create summary statistic cards for the dashboard."""
    total_vehicles = len(df)
    avg_lead_time = df['New_Lead_Time_Days'].mean()
    median_lead_time = df['New_Lead_Time_Days'].median()
    max_lead_time = df['New_Lead_Time_Days'].max()
    problematic_vehicles = len(df[df['New_Lead_Time_Days'] > 45])
    
    top_10_percent_threshold = df['New_Lead_Time_Days'].quantile(0.9)
    optimized_df = df[df['New_Lead_Time_Days'] <= top_10_percent_threshold]
    optimized_avg = optimized_df['New_Lead_Time_Days'].mean()
    reduction_potential = ((avg_lead_time - optimized_avg) /
                           avg_lead_time) * 100
    
    cards = [
        {
            'title': 'Total Vehicles Analyzed',
            'value': f"{total_vehicles:,}",
            'subtitle': 'Across all vehicle types'
        },
        {
            'title': 'Average Lead Time',
            'value': f"{avg_lead_time:.1f} days",
            'subtitle': 'From part production to assembly'
        },
        {
            'title': 'Median Lead Time',
            'value': f"{median_lead_time:.1f} days",
            'subtitle': 'Less sensitive to outliers'
        },
        {
            'title': 'Maximum Lead Time',
            'value': f"{max_lead_time:.0f} days",
            'subtitle': 'Longest production cycle observed'
        }
    ]
    
    return [
        html.Div([
            html.H3(card['value'],
                    style={'color': CORPORATE_COLORS['secondary'],
                           'margin': '0',
                           'fontSize': '2rem',
                           'fontWeight': '700'}),
            html.H4(card['title'],
                    style={'color': CORPORATE_COLORS['dark'],
                           'margin': '5px 0',
                           'fontSize': '1rem',
                           'fontWeight': '600'}),
            html.P(card['subtitle'],
                   style={'color': CORPORATE_COLORS['dark'],
                          'margin': '0',
                          'fontSize': '0.85rem',
                          'opacity': '0.8'})
        ], style={
            'backgroundColor': CORPORATE_COLORS['white'],
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
            'textAlign': 'center',
            'border': f'1px solid {CORPORATE_COLORS["accent"]}',
            'height': '120px',
            'display': 'flex',
            'flexDirection': 'column',
            'justifyContent': 'center'
        }) for card in cards
    ]

# Load data
df = load_and_prepare_data()
logo_encoded = encode_logo()

# Application Layout
app.layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Img(src=logo_encoded,
                        style={'height': '60px',
                               'marginRight': '20px'}) if logo_encoded else None
            ], style={'display': 'flex', 'alignItems': 'center'}),
            
            html.Div([
                html.H1("Automotive Production Lead Time Analysis",
                       style={'color': CORPORATE_COLORS['white'],
                              'margin': '0',
                              'fontSize': '2.2rem',
                              'fontWeight': '700'}),
                html.P("Identifying Vehicle Types with Extended Production Cycles",
                      style={'color': CORPORATE_COLORS['accent'],
                             'margin': '5px 0 0 0',
                             'fontSize': '1.1rem'})
            ], style={'flex': '1'})
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'maxWidth': '1200px',
            'margin': '0 auto',
            'padding': '0 20px'
        })
    ], style={
        'backgroundColor': CORPORATE_COLORS['secondary'],
        'padding': '20px 0',
        'marginBottom': '30px'
    }),
    
    html.Div([
        html.Div([
            html.H2("Executive Summary",
                   style={'color': CORPORATE_COLORS['dark'],
                          'marginBottom': '20px',
                          'fontSize': '1.5rem',
                          'fontWeight': '600'}),
            html.Div(create_summary_cards(df),
                    style={'display': 'grid',
                           'gridTemplateColumns': 'repeat(auto-fit, minmax(250px, 1fr))',
                           'gap': '20px'})
        ], style={'marginBottom': '40px'}),
        
        html.Div([
            html.H2("Analysis Controls",
                   style={'color': CORPORATE_COLORS['dark'],
                          'marginBottom': '20px',
                          'fontSize': '1.5rem',
                          'fontWeight': '600'}),
            html.Div([
                html.Div([
                    html.Label("Select Vehicle Type(s):",
                             style={'fontWeight': '600',
                                    'color': CORPORATE_COLORS['dark'],
                                    'marginBottom': '8px',
                                    'display': 'block'}),
                    dcc.Dropdown(
                        id='vehicle-type-dropdown',
                        options=[{'label': vt, 'value': vt}
                                for vt in sorted(df['Vehicle_Type'].unique())],
                        value=sorted(df['Vehicle_Type'].unique()),
                        multi=True,
                        style={'fontFamily': FONT_FAMILY}
                    )
                ], style={'width': '48%', 'marginRight': '4%'}),
                
                html.Div([
                    html.Label("Lead Time Range (days):",
                             style={'fontWeight': '600',
                                    'color': CORPORATE_COLORS['dark'],
                                    'marginBottom': '8px',
                                    'display': 'block'}),
                    dcc.RangeSlider(
                        id='leadtime-range-slider',
                        min=df['New_Lead_Time_Days'].min(),
                        max=df['New_Lead_Time_Days'].max(),
                        step=1,
                        marks={i: f'{i}d' for i in range(
                            int(df['New_Lead_Time_Days'].min()),
                            int(df['New_Lead_Time_Days'].max())+1, 10)},
                        value=[df['New_Lead_Time_Days'].min(), df['New_Lead_Time_Days'].max()],
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'width': '48%'})
            ], style={'display': 'flex', 'alignItems': 'flex-end'})
        ], style={
            'backgroundColor': CORPORATE_COLORS['light'],
            'padding': '20px',
            'borderRadius': '8px',
            'marginBottom': '30px'
        }),
        
        # Main Analysis Charts
        html.Div([
            html.Div([
                html.H3("Lead Time Distribution by Vehicle Type",
                       style={'color': CORPORATE_COLORS['dark'],
                              'marginBottom': '15px',
                              'fontSize': '1.3rem',
                              'fontWeight': '600'}),
                dcc.Graph(id='leadtime-distribution-chart'),
                html.Div(id='leadtime-summary', style={
                    'marginTop': '20px',
                    'padding': '15px',
                    'backgroundColor': CORPORATE_COLORS['light'],
                    'borderRadius': '5px',
                    'border': f'1px solid {CORPORATE_COLORS["accent"]}'
                })
            ], style={
                'backgroundColor': CORPORATE_COLORS['white'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            }),
            
            html.Div([
                html.H3("Component Lead Time Analysis",
                       style={'color': CORPORATE_COLORS['dark'],
                              'marginBottom': '15px',
                              'fontSize': '1.3rem',
                              'fontWeight': '600'}),
                dcc.Graph(id='component-analysis-chart'),
                html.Div(id='component-summary', style={
                    'marginTop': '20px',
                    'padding': '15px',
                    'backgroundColor': CORPORATE_COLORS['light'],
                    'borderRadius': '5px',
                    'border': f'1px solid {CORPORATE_COLORS["accent"]}'
                })
            ], style={
                'backgroundColor': CORPORATE_COLORS['white'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            }),

            # Part Type Delay Analysis
            html.Div([
                html.H3("Part Type Delay Analysis",
                       style={'color': CORPORATE_COLORS['dark'],
                              'marginBottom': '15px',
                              'fontSize': '1.3rem',
                              'fontWeight': '600'}),
                html.P("Analysis of which part types (1-40) cause the most production delays by becoming the bottleneck (latest) part.",
                       style={'color': CORPORATE_COLORS['dark'],
                              'marginBottom': '15px',
                              'fontSize': '0.9rem'}),
                dcc.Graph(id='part-type-delay-chart'),
                html.Div(id='part-type-delay-summary', style={
                    'marginTop': '20px',
                    'padding': '15px',
                    'backgroundColor': CORPORATE_COLORS['light'],
                    'borderRadius': '5px',
                    'border': f'1px solid {CORPORATE_COLORS["accent"]}'
                })
            ], style={
                'backgroundColor': CORPORATE_COLORS['white'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            })
        ]),
        
        html.Div([
            html.H3("Production Pattern Analysis by Vehicle Type",
                   style={'color': CORPORATE_COLORS['dark'],
                          'marginBottom': '15px',
                          'fontSize': '1.3rem',
                          'fontWeight': '600'}),
            
            # Main content area with relative positioning
            html.Div([
                dcc.Graph(id='production-pattern-chart'),
                
                # Toggle buttons positioned in bottom right corner
                html.Div([
                    dcc.RadioItems(
                        id='pie-chart-display-mode',
                        options=[
                            {'label': 'Percentage', 'value': 'percentage'},
                            {'label': 'Absolute Numbers', 'value': 'absolute'}
                        ],
                        value='percentage',
                        inline=True,
                        style={'marginBottom': '0'}
                    )
                ], style={
                    'position': 'absolute',
                    'bottom': '10px',
                    'right': '10px',
                    'padding': '8px 12px',
                    'backgroundColor': 'rgba(255, 255, 255, 0.9)',
                    'borderRadius': '5px',
                    'border': f'1px solid {CORPORATE_COLORS["secondary"]}',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'fontSize': '0.9rem'
                })
            ], style={'position': 'relative'}),
            
            html.Div(id='production-pattern-summary', style={
                'marginTop': '20px',
                'padding': '15px',
                'backgroundColor': CORPORATE_COLORS['light'],
                'borderRadius': '5px',
                'border': f'1px solid {CORPORATE_COLORS["accent"]}'
            })
        ], style={
            'backgroundColor': CORPORATE_COLORS['white'],
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
            'marginBottom': '20px'
        }),
        
            html.Div([
                html.H3("Factory Lead Time Distribution by Component",
                       style={'color': CORPORATE_COLORS['dark'],
                              'marginBottom': '15px',
                              'fontSize': '1.3rem',
                              'fontWeight': '600'}),
                html.P("Distribution of the longest average lead times by factory (plant) for each component. The first 3 digits of the plant code indicate the manufacturer.",
                      style={'color': CORPORATE_COLORS['dark'],
                             'marginBottom': '15px',
                             'fontSize': '0.9rem',
                             'fontStyle': 'italic'}),
                dcc.Graph(id='factory-leadtime-distribution-chart'),
                html.Div(id='factory-leadtime-summary', style={
                    'marginTop': '20px',
                    'padding': '15px',
                    'backgroundColor': CORPORATE_COLORS['light'],
                    'borderRadius': '5px',
                    'border': f'1px solid {CORPORATE_COLORS["accent"]}'
                })
            ], style={
                'backgroundColor': CORPORATE_COLORS['white'],
                'padding': '20px',
                'borderRadius': '8px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                'marginBottom': '20px'
            })
        
    ], style={
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '0 20px'
    }),
    
    # High Lead Time Vehicles Table Section
    html.Div([
        html.H2("Critical Production Cases",
                style={'color': CORPORATE_COLORS['dark'],
                       'marginBottom': '20px',
                       'fontSize': '1.8rem',
                       'fontWeight': '700',
                       'textAlign': 'center'}),
        
        html.P("Top 100 vehicles with highest total component lead times requiring immediate attention",
               style={'color': CORPORATE_COLORS['dark'],
                      'textAlign': 'center',
                      'marginBottom': '30px',
                      'fontSize': '1.1rem',
                      'fontStyle': 'italic'}),
        
        html.Div([
            dash_table.DataTable(
                id='high-leadtime-vehicles-table',
                columns=[],
                data=[],
                style_cell={
                    'textAlign': 'left',
                    'fontFamily': FONT_FAMILY,
                    'fontSize': '0.9rem',
                    'padding': '8px'
                },
                style_header={
                    'backgroundColor': CORPORATE_COLORS['primary'],
                    'color': 'white',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': CORPORATE_COLORS['light']
                    }
                ],
                page_size=25,
                sort_action='native'
            )
        ], style={
            'backgroundColor': 'white',
            'padding': '20px',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'marginBottom': '20px'
        })
        
    ], style={
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '0 20px'
    }),
    
    html.Div([
        html.P(("Automotive Production Analysis Dashboard | "
               "Case Study IDA Group 40"),
              style={'color': CORPORATE_COLORS['white'],
                     'textAlign': 'center',
                     'margin': '0',
                     'fontSize': '0.9rem'})
    ], style={
        'backgroundColor': CORPORATE_COLORS['dark'],
        'padding': '15px 0',
        'marginTop': '40px'
    })
], style={
    'fontFamily': FONT_FAMILY,
    'backgroundColor': CORPORATE_COLORS['light'],
    'minHeight': '100vh'
})

# Lead Time Distribution Chart Callback
@app.callback(
    Output('leadtime-distribution-chart', 'figure'),
    [Input('vehicle-type-dropdown', 'value'),
     Input('leadtime-range-slider', 'value')]
)
def update_leadtime_distribution(selected_types, leadtime_range):
    # Filter using new lead time metric
    filtered_df = df[
        (df['Vehicle_Type'].isin(selected_types)) &
        (df['New_Lead_Time_Days'] >= leadtime_range[0]) &
        (df['New_Lead_Time_Days'] <= leadtime_range[1])
    ]

    # Get all vehicle types from filtered data
    vehicle_types = sorted(filtered_df['Vehicle_Type'].unique())

    # Calculate 95th percentile threshold per vehicle type
    vehicle_thresholds = {}
    main_data_all = []
    outlier_data_all = []
    
    for vtype in vehicle_types:
        vtype_data = filtered_df[filtered_df['Vehicle_Type'] == vtype]
        threshold_95 = vtype_data['New_Lead_Time_Days'].quantile(0.95)
        vehicle_thresholds[vtype] = threshold_95
        
        vtype_main = vtype_data[vtype_data['New_Lead_Time_Days'] <= threshold_95]
        vtype_outliers = vtype_data[vtype_data['New_Lead_Time_Days'] > threshold_95]
        
        main_data_all.append(vtype_main)
        outlier_data_all.append(vtype_outliers)
    
    main_data = pd.concat(main_data_all, ignore_index=True) if main_data_all else pd.DataFrame()
    outlier_data = pd.concat(outlier_data_all, ignore_index=True) if outlier_data_all else pd.DataFrame()
    
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=('Lead Time Distribution (95% of data per type)', 'Maximum Delay Days per Vehicle Type'),
        horizontal_spacing=0.1
    )
    
    colors = [CORPORATE_COLORS['primary'], CORPORATE_COLORS['secondary'], 
              CORPORATE_COLORS['accent'], CORPORATE_COLORS['warning']]
    
    for i, vtype in enumerate(vehicle_types):
        vtype_data = main_data[main_data['Vehicle_Type'] == vtype]
        fig.add_trace(
            go.Violin(
                y=vtype_data['New_Lead_Time_Days'],
                x=[vtype] * len(vtype_data),
                name=vtype,
                box_visible=True,
                meanline_visible=True,
                fillcolor=colors[i % len(colors)],
                opacity=0.7,
                line_color=colors[i % len(colors)],
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Calculate outlier statistics per vehicle type
    delay_counts = {}
    max_delays = {}
    for vtype in vehicle_types:
        vtype_outliers = outlier_data[outlier_data['Vehicle_Type'] == vtype]
        delay_counts[vtype] = len(vtype_outliers)
        if not vtype_outliers.empty:
            max_delays[vtype] = vtype_outliers['New_Lead_Time_Days'].max()
        else:
            max_delays[vtype] = 0

    fig.add_trace(
        go.Bar(
            x=list(max_delays.keys()),
            y=list(max_delays.values()),
            name='Maximum Delay Days',
            marker_color=colors[:len(delay_counts)],
            opacity=0.8,
            showlegend=False,
            text=[f"{max_delays[vtype]:.0f}d" if max_delays[vtype] > 0 else "0d" for vtype in max_delays.keys()],
            textposition='inside',
            textfont=dict(color='white', size=12, family=FONT_FAMILY)
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family=FONT_FAMILY, color=CORPORATE_COLORS['dark']),
        showlegend=False,
        height=500,
        title_text=f"Lead Time Analysis (95th percentile per type)"
    )
    
    fig.update_yaxes(title_text="Lead Time (days)", row=1, col=1)
    fig.update_xaxes(title_text="Vehicle Type", row=1, col=1, tickangle=45)
    
    fig.update_yaxes(title_text="Maximum Delay (days)", row=1, col=2)
    fig.update_xaxes(title_text="Vehicle Type", row=1, col=2, tickangle=45)
    
    return fig

# Lead Time Distribution Summary Callback
@app.callback(
    Output('leadtime-summary', 'children'),
    [Input('vehicle-type-dropdown', 'value'),
     Input('leadtime-range-slider', 'value')]
)
def update_leadtime_summary(selected_types, leadtime_range):
    filtered_df = df[
        (df['Vehicle_Type'].isin(selected_types)) &
        (df['New_Lead_Time_Days'] >= leadtime_range[0]) &
        (df['New_Lead_Time_Days'] <= leadtime_range[1])
    ]
    
    threshold_95 = filtered_df['New_Lead_Time_Days'].quantile(0.95)
    main_data = filtered_df[filtered_df['New_Lead_Time_Days'] <= threshold_95]
    outlier_data = filtered_df[filtered_df['New_Lead_Time_Days'] > threshold_95]
    
    stats = main_data.groupby('Vehicle_Type')['New_Lead_Time_Days'].agg(['mean', 'median', 'std', 'count']).round(2)
    
    best_performer = stats['mean'].idxmin()
    worst_performer = stats['mean'].idxmax()
    
    outlier_count = len(outlier_data)
    total_count = len(filtered_df)
    outlier_percentage = (outlier_count / total_count * 100) if total_count > 0 else 0
    
    oem1_data = filtered_df[filtered_df['Vehicle_Type'].str.contains('OEM1')]
    oem2_data = filtered_df[filtered_df['Vehicle_Type'].str.contains('OEM2')]
    
    oem1_avg = oem1_data['New_Lead_Time_Days'].mean() if not oem1_data.empty else 0
    oem2_avg = oem2_data['New_Lead_Time_Days'].mean() if not oem2_data.empty else 0
    oem1_count = len(oem1_data)
    oem2_count = len(oem2_data)
    
    summary_text = [
        html.H4("Lead Time Distribution Analysis Summary",
                style={'color': CORPORATE_COLORS['dark'],
                       'marginBottom': '10px',
                       'fontSize': '1.1rem',
                       'fontWeight': '600'}),
        html.P([
            f"Violin plot analysis of {total_count} vehicles with 95th percentile threshold. ",
            f"Lead time calculated as Vehicle Production Date minus Latest Component Production Date. ",
            f"Main distribution (95% of data) shows ", html.Strong(f"{best_performer}"), f" achieves optimal performance with {stats.loc[best_performer, 'mean']:.1f} days average, ",
            f"while ", html.Strong(f"{worst_performer}"), f" requires {stats.loc[worst_performer, 'mean']:.1f} days on average. ",
            f"Manufacturer comparison reveals ", html.Strong("OEM1"), f" produces {oem1_count} vehicles with {oem1_avg:.1f} days average lead time, ",
            f"while ", html.Strong("OEM2"), f" manufactures {oem2_count} vehicles averaging {oem2_avg:.1f} days, demonstrating insignificant operational efficiency profiles between manufacturers."
        ], style={'color': CORPORATE_COLORS['dark'], 'fontSize': '1rem', 'lineHeight': '1.6', 'margin': '0'})
    ]
    
    return summary_text

# Component Analysis Chart Callback
@app.callback(
    Output('component-analysis-chart', 'figure'),
    [Input('vehicle-type-dropdown', 'value'),
     Input('leadtime-range-slider', 'value')]
)
def update_component_analysis(selected_types, leadtime_range):
    filtered_df = df[
        (df['Vehicle_Type'].isin(selected_types)) &
        (df['New_Lead_Time_Days'] >= leadtime_range[0]) &
        (df['New_Lead_Time_Days'] <= leadtime_range[1])
    ]
    
    # Reshape data for component analysis
    component_data = []
    for _, row in filtered_df.iterrows():
        if pd.notna(row['New_Karosserie_LeadTime']):
            component_data.append({'Component': 'Body', 'Lead_Time': row['New_Karosserie_LeadTime']})
        if pd.notna(row['New_Schaltung_LeadTime']):
            component_data.append({'Component': 'Transmission', 'Lead_Time': row['New_Schaltung_LeadTime']})
        if pd.notna(row['New_Sitze_LeadTime']):
            component_data.append({'Component': 'Seats', 'Lead_Time': row['New_Sitze_LeadTime']})
        if pd.notna(row['New_Motor_LeadTime']):
            component_data.append({'Component': 'Engine', 'Lead_Time': row['New_Motor_LeadTime']})

    component_df = pd.DataFrame(component_data)

    if component_df.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No component data available for selected filters",
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    
    # Calculate 95th percentile threshold per component type
    components = ['Body', 'Transmission', 'Seats', 'Engine']
    component_thresholds = {}
    main_data_all = []
    outlier_data_all = []
    
    for component in components:
        comp_data = component_df[component_df['Component'] == component]
        if not comp_data.empty:
            threshold_95 = comp_data['Lead_Time'].quantile(0.95)
            component_thresholds[component] = threshold_95
            
            comp_main = comp_data[comp_data['Lead_Time'] <= threshold_95]
            comp_outliers = comp_data[comp_data['Lead_Time'] > threshold_95]
            
            main_data_all.append(comp_main)
            outlier_data_all.append(comp_outliers)
        else:
            component_thresholds[component] = 0
    
    main_data = pd.concat(main_data_all, ignore_index=True) if main_data_all else pd.DataFrame()
    outlier_data = pd.concat(outlier_data_all, ignore_index=True) if outlier_data_all else pd.DataFrame()
    
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.7, 0.3],
        subplot_titles=('Component Lead Time Distribution (95% of data per type)', 'Maximum Delay Days per Component'),
        horizontal_spacing=0.1
    )
    
    colors = [CORPORATE_COLORS['primary'], CORPORATE_COLORS['secondary'], 
              CORPORATE_COLORS['accent'], CORPORATE_COLORS['warning']]
    
    for i, component in enumerate(components):
        comp_data = main_data[main_data['Component'] == component]
        if not comp_data.empty:
            fig.add_trace(
                go.Violin(
                    y=comp_data['Lead_Time'],
                    x=[component] * len(comp_data),
                    name=component,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=colors[i],
                    opacity=0.7,
                    line_color=colors[i],
                    showlegend=False
                ),
                row=1, col=1
            )
    
    # Add bar chart for outliers (5%) - show count of items with long delays for each component
    # Calculate count of items with long delays and maximum delay for each component
    delay_counts = {}
    max_delays = {}
    for component in components:
        comp_outliers = outlier_data[outlier_data['Component'] == component]
        delay_counts[component] = len(comp_outliers)
        if not comp_outliers.empty:
            max_delays[component] = comp_outliers['Lead_Time'].max()
        else:
            max_delays[component] = 0
    
    # Create bar chart with all components
    fig.add_trace(
        go.Bar(
            x=list(max_delays.keys()),
            y=list(max_delays.values()),
            name='Maximum Delay Days',
            marker_color=colors[:len(delay_counts)],
            opacity=0.8,
            showlegend=False,
            text=[f"{max_delays[comp]:.0f}d" if max_delays[comp] > 0 else "0d" for comp in max_delays.keys()],
            textposition='inside',
            textfont=dict(color='white', size=12, family=FONT_FAMILY)
        ),
        row=1, col=2
    )
    
    # Create title showing individual thresholds
    threshold_text = ", ".join([f"{comp}: {component_thresholds[comp]:.1f}d" for comp in components if component_thresholds[comp] > 0])
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family=FONT_FAMILY, color=CORPORATE_COLORS['dark']),
        showlegend=False,
        height=500,
        title_text=f"Component Lead Time Analysis (95th percentile per type - {threshold_text})"
    )
    
    fig.update_yaxes(title_text="Lead Time (days)", row=1, col=1)
    fig.update_xaxes(title_text="Component Type", row=1, col=1)
    
    fig.update_yaxes(title_text="Maximum Delay (days)", row=1, col=2)
    fig.update_xaxes(title_text="Component Type", row=1, col=2)
    
    return fig

# Component Analysis Summary Callback
@app.callback(
    Output('component-summary', 'children'),
    [Input('vehicle-type-dropdown', 'value'),
     Input('leadtime-range-slider', 'value')]
)
def update_component_summary(selected_types, leadtime_range):
    filtered_df = df[
        (df['Vehicle_Type'].isin(selected_types)) &
        (df['New_Lead_Time_Days'] >= leadtime_range[0]) &
        (df['New_Lead_Time_Days'] <= leadtime_range[1])
    ]
    
    # Prepare component data for analysis
    component_data = []
    component_columns = {
        'Body': 'New_Karosserie_LeadTime',
        'Transmission': 'New_Schaltung_LeadTime', 
        'Seats': 'New_Sitze_LeadTime',
        'Engine': 'New_Motor_LeadTime'
    }
    
    for comp_name, col_name in component_columns.items():
        comp_values = filtered_df[col_name].dropna()
        for value in comp_values:
            component_data.append({'Component': comp_name, 'Lead_Time': value})
    
    if not component_data:
        return [html.P("No component data available for selected filters.")]
    
    component_df = pd.DataFrame(component_data)
    
    # Calculate 95th percentile threshold
    threshold_95 = component_df['Lead_Time'].quantile(0.95)
    main_data = component_df[component_df['Lead_Time'] <= threshold_95]
    outlier_data = component_df[component_df['Lead_Time'] > threshold_95]
    
    # Calculate statistics for main data
    main_stats = main_data.groupby('Component')['Lead_Time'].agg(['mean', 'std', 'count']).round(2)
    
    # Find performance leaders
    best_component = main_stats['mean'].idxmin()
    worst_component = main_stats['mean'].idxmax()
    most_variable = (main_stats['std'] / main_stats['mean']).idxmax()
    
    # Outlier analysis
    outlier_counts = outlier_data['Component'].value_counts() if not outlier_data.empty else pd.Series()
    total_outliers = len(outlier_data)
    total_components = len(component_df)
    outlier_percentage = (total_outliers / total_components * 100) if total_components > 0 else 0
    
    summary_text = [
        html.H4("Component Lead Time Analysis Summary",
                style={'color': CORPORATE_COLORS['dark'],
                       'marginBottom': '10px',
                       'fontSize': '1.1rem',
                       'fontWeight': '600'}),
        html.P([
            f"The visualization displays lead time performance across {total_components:,} component measurements using two complementary views. ",
            f"The left violin plot shows the distribution shape and density of lead times for each component type, revealing typical performance ranges and outlier patterns. ",
            f"The right bar chart displays maximum delay extremes, highlighting worst-case scenarios that impact production planning. ",
            f"Analysis reveals ", html.Strong(f"{best_component}"), f" components demonstrate the most consistent performance with an average of {main_stats.loc[best_component, 'mean']:.1f} days, ",
            f"while ", html.Strong(f"{worst_component}"), f" components show extended timelines averaging {main_stats.loc[worst_component, 'mean']:.1f} days. ",
            f"The 95th percentile threshold of {threshold_95:.1f} days helps identify {outlier_percentage:.1f}% of cases requiring management attention for supply chain optimization."
        ], style={'color': CORPORATE_COLORS['dark'], 'fontSize': '1rem', 'lineHeight': '1.6', 'margin': '0'})
    ]
    
    return summary_text

# Part Type Delay Analysis Chart Callback
@app.callback(
    Output('part-type-delay-chart', 'figure'),
    [Input('vehicle-type-dropdown', 'value'),
     Input('leadtime-range-slider', 'value')]
)
def update_part_type_delay_analysis(selected_types, leadtime_range):
    filtered_df = df[
        (df['Vehicle_Type'].isin(selected_types)) &
        (df['New_Lead_Time_Days'] >= leadtime_range[0]) &
        (df['New_Lead_Time_Days'] <= leadtime_range[1])
    ].copy()
    
    # Extract part types from latest part IDs (first number before the first dash)
    latest_id_cols = ['Karosserie_Latest_Part_ID', 'Schaltung_Latest_Part_ID', 
                      'Sitze_Latest_Part_ID', 'Motor_Latest_Part_ID']
    component_names = ['Body', 'Transmission', 'Seats', 'Engine']
    
    # Collect part type data for each component
    part_type_data = []
    
    for col, component in zip(latest_id_cols, component_names):
        if col in filtered_df.columns:
            # Extract part type (first number) from part ID
            part_types = filtered_df[col].astype(str).str.split('-').str[0].astype(int)
            part_type_counts = part_types.value_counts()
            
            for part_type, count in part_type_counts.items():
                part_type_data.append({
                    'Part_Type': part_type,
                    'Component': component,
                    'Delay_Count': count
                })
    
    if not part_type_data:
        fig = go.Figure()
        fig.update_layout(title='No part type delay data available')
        return fig

    # Create DataFrame from collected data
    part_type_df = pd.DataFrame(part_type_data)    # Get part types with actual delays (remove empty columns)
    part_type_totals = (part_type_df.groupby('Part_Type')['Delay_Count']
                       .sum().sort_values(ascending=False))
    
    # Filter out part types with zero delays and get top 15 for better readability
    non_zero_part_types = part_type_totals[part_type_totals > 0].head(15)
    
    # Filter data to non-zero part types only
    filtered_part_type_df = part_type_df[part_type_df['Part_Type'].isin(non_zero_part_types.index)]
    
    # Create part type name mapping for better labels
    def get_part_type_name(part_type):
        """Generate descriptive names for part types"""
        type_ranges = {
            # Engine parts (1-10)
            range(1, 11): "Engine",
            # Seats parts (11-20) 
            range(11, 21): "Seats",
            # Transmission parts (21-30)
            range(21, 31): "Transmission", 
            # Body parts (31-40)
            range(31, 41): "Body"
        }
        
        for range_obj, component in type_ranges.items():
            if part_type in range_obj:
                return f"{component} Part {part_type}"
        return f"Part Type {part_type}"
    
    # Create x-axis labels with part type names
    x_labels = [get_part_type_name(pt) for pt in non_zero_part_types.index]
    
    # Create stacked bar chart
    fig = go.Figure()
    
    # Component colors
    component_colors = {
        'Body': CORPORATE_COLORS['primary'],
        'Engine': CORPORATE_COLORS['secondary'], 
        'Transmission': CORPORATE_COLORS['accent'],
        'Seats': CORPORATE_COLORS['warning']
    }
    
    # Add bars for each component (only show components that have data)
    for component in component_names:
        component_data = filtered_part_type_df[filtered_part_type_df['Component'] == component]
        if not component_data.empty:
            # Create full series for all part types (fill missing with 0)
            part_type_series = component_data.set_index('Part_Type')['Delay_Count']
            part_type_series = part_type_series.reindex(non_zero_part_types.index, fill_value=0)
            
            # Only add trace if this component has any non-zero values
            if part_type_series.sum() > 0:
                fig.add_trace(go.Bar(
                    x=x_labels,
                    y=part_type_series.values,
                    name=component,
                    marker_color=component_colors.get(component, CORPORATE_COLORS['primary']),
                    hovertemplate=f'<b>{component}</b><br>' +
                                 'Part: %{x}<br>' +
                                 'Delay Count: %{y}<br>' +
                                 '<extra></extra>'
                ))
    
    fig.update_layout(
        title='Top 15 Part Types Causing Most Production Delays',
        xaxis_title='Part Type',
        yaxis_title='Number of Times as Bottleneck Part',
        barmode='stack',
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family=FONT_FAMILY, color=CORPORATE_COLORS['dark']),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        xaxis=dict(
            tickangle=45,
            tickmode='array',
            tickvals=list(range(len(x_labels))),
            ticktext=x_labels
        )
    )
    
    return fig

# Part Type Delay Analysis Summary Callback
@app.callback(
    Output('part-type-delay-summary', 'children'),
    [Input('vehicle-type-dropdown', 'value'),
     Input('leadtime-range-slider', 'value')]
)
def update_part_type_delay_summary(selected_types, leadtime_range):
    filtered_df = df[
        (df['Vehicle_Type'].isin(selected_types)) &
        (df['New_Lead_Time_Days'] >= leadtime_range[0]) &
        (df['New_Lead_Time_Days'] <= leadtime_range[1])
    ].copy()
    
    # Extract part types from latest part IDs
    latest_id_cols = ['Karosserie_Latest_Part_ID', 'Schaltung_Latest_Part_ID', 
                      'Sitze_Latest_Part_ID', 'Motor_Latest_Part_ID']
    component_names = ['Body', 'Transmission', 'Seats', 'Engine']
    
    # Collect overall statistics
    all_part_types = []
    component_stats = {}
    
    for col, component in zip(latest_id_cols, component_names):
        if col in filtered_df.columns:
            part_types = filtered_df[col].astype(str).str.split('-').str[0].astype(int)
            part_type_counts = part_types.value_counts()
            all_part_types.extend(part_types.tolist())
            
            # Component-specific stats
            component_stats[component] = {
                'unique_types': len(part_type_counts),
                'most_problematic': part_type_counts.index[0] if not part_type_counts.empty else None,
                'most_problematic_count': part_type_counts.iloc[0] if not part_type_counts.empty else 0
            }
    
    if not all_part_types:
        return html.P("No part type data available for analysis.", 
                     style={'color': CORPORATE_COLORS['dark']})
    
    # Overall statistics
    all_part_types_series = pd.Series(all_part_types)
    overall_counts = all_part_types_series.value_counts()
    most_problematic_overall = overall_counts.index[0]
    total_unique_types = len(overall_counts)
    
    # Get part type name for display
    def get_part_type_name(part_type):
        """Generate descriptive names for part types"""
        type_ranges = {
            range(1, 11): "Engine",
            range(11, 21): "Seats",
            range(21, 31): "Transmission", 
            range(31, 41): "Body"
        }
        
        for range_obj, component in type_ranges.items():
            if part_type in range_obj:
                return f"{component} Part {part_type}"
        return f"Part Type {part_type}"
    
    # Find component with most diverse part types
    most_diverse_component = max(component_stats.keys(), 
                                key=lambda x: component_stats[x]['unique_types'])
    
    # Find component with most problematic single part type
    most_problematic_component = max(component_stats.keys(), 
                                   key=lambda x: component_stats[x]['most_problematic_count'])
    
    # Calculate active vs total part types
    active_part_types = len([count for count in overall_counts if count > 0])
    
    summary_text = [
        html.H4("Part Type Delay Analysis Summary",
                style={'color': CORPORATE_COLORS['dark'],
                       'marginBottom': '10px',
                       'fontSize': '1.1rem',
                       'fontWeight': '600'}),
        html.P([
            f"Analysis of {active_part_types} active part types across {len(filtered_df)} vehicles showing actual bottleneck occurrences. ",
            html.Strong(f"{get_part_type_name(most_problematic_overall)}"), f" causes the most delays overall, appearing as the bottleneck part {overall_counts.iloc[0]} times. ",
            html.Strong(f"{most_diverse_component}"), f" components utilize the most diverse part types ({component_stats[most_diverse_component]['unique_types']} different types). ",
            f"Chart displays only part types with actual delay occurrences, filtered from potential range 1-40."
        ], style={'color': CORPORATE_COLORS['dark'], 'fontSize': '1rem', 'lineHeight': '1.6', 'margin': '0'})
    ]
    
    return summary_text


# Production Pattern Analysis Chart Callback
@app.callback(
    Output('production-pattern-chart', 'figure'),
    [Input('vehicle-type-dropdown', 'value'),
     Input('leadtime-range-slider', 'value'),
     Input('pie-chart-display-mode', 'value')]
)
def update_production_pattern_analysis(selected_types, leadtime_range, display_mode):
    filtered_df = df[
        (df['Vehicle_Type'].isin(selected_types)) &
        (df['New_Lead_Time_Days'] >= leadtime_range[0]) &
        (df['New_Lead_Time_Days'] <= leadtime_range[1])
    ].copy()
    
    # Create a 2x2 subplot layout for comprehensive production analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Production Volume', 'Production Efficiency Trends',
                       'Seasonal Production Patterns', 'Vehicle Type Production Share'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Chart 1: Monthly Production Volume by Vehicle Type
    monthly_production = filtered_df.groupby(['Production_Year', 'Production_Month', 'Vehicle_Type']).size().reset_index(name='Count')
    monthly_production['Date'] = pd.to_datetime({
        'year': monthly_production['Production_Year'],
        'month': monthly_production['Production_Month'],
        'day': 1
    })
    
    vehicle_colors = {
        'OEM1_TYP11': CORPORATE_COLORS['primary'],
        'OEM1_TYP12': CORPORATE_COLORS['secondary'],
        'OEM2_TYP21': CORPORATE_COLORS['accent'],
        'OEM2_TYP22': CORPORATE_COLORS['warning']
    }
    
    for vehicle_type in selected_types:
        type_data = monthly_production[monthly_production['Vehicle_Type'] == vehicle_type]
        if not type_data.empty:
            fig.add_trace(go.Scatter(
                x=type_data['Date'],
                y=type_data['Count'],
                mode='lines+markers',
                name=vehicle_type,
                line=dict(color=vehicle_colors.get(vehicle_type, CORPORATE_COLORS['primary']), width=2),
                marker=dict(size=4),
                showlegend=False, 
                hovertemplate=f'<b>{vehicle_type}</b><br>' +
                             'Date: %{x}<br>' +
                             'Production: %{y} vehicles<extra></extra>'
            ), row=1, col=1)
    
    # Chart 2: Production Efficiency Trends (Average Lead Time Over Time)
    monthly_efficiency = filtered_df.groupby(['Production_Year', 'Production_Month', 'Vehicle_Type'])['New_Lead_Time_Days'].mean().reset_index()
    monthly_efficiency['Date'] = pd.to_datetime({
        'year': monthly_efficiency['Production_Year'],
        'month': monthly_efficiency['Production_Month'],
        'day': 1
    })
    
    for vehicle_type in selected_types:
        type_data = monthly_efficiency[monthly_efficiency['Vehicle_Type'] == vehicle_type]
        if not type_data.empty:
            fig.add_trace(go.Scatter(
                x=type_data['Date'],
                y=type_data['New_Lead_Time_Days'],
                mode='lines+markers',
                name=f'{vehicle_type} Efficiency',
                line=dict(color=vehicle_colors.get(vehicle_type, CORPORATE_COLORS['primary']), width=2, dash='dot'),
                marker=dict(size=4),
                showlegend=False,
                hovertemplate=f'<b>{vehicle_type} Lead Time</b><br>' +
                             'Date: %{x}<br>' +
                             'Avg Lead Time: %{y:.1f} days<extra></extra>'
            ), row=1, col=2)
    
    # Chart 3: Seasonal Production Patterns (Stacked by Vehicle Type)
    seasonal_data = filtered_df.groupby(['Production_Month', 'Vehicle_Type']).size().reset_index(name='Count')
    
    # Add stacked bars for each vehicle type
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for vehicle_type in selected_types:
        type_seasonal = seasonal_data[seasonal_data['Vehicle_Type'] == vehicle_type]
        if not type_seasonal.empty:
            # Create full month range (1-12) with zeros for missing months
            months = list(range(1, 13))
            counts = []
            for month in months:
                month_data = type_seasonal[type_seasonal['Production_Month'] == month]
                counts.append(month_data['Count'].iloc[0] if not month_data.empty else 0)
            
            fig.add_trace(go.Bar(
                x=month_names,
                y=counts,
                name=vehicle_type,
                marker_color=vehicle_colors.get(vehicle_type, CORPORATE_COLORS['primary']),
                opacity=0.8,
                showlegend=True,
                legendgroup=vehicle_type, 
                hovertemplate=f'<b>{vehicle_type}</b><br>' +
                             'Month: %{x}<br>' +
                             'Production: %{y} vehicles<extra></extra>'
            ), row=2, col=1)
    
    # Chart 4: Vehicle Type Production Share (Pie Chart)
    production_share = filtered_df['Vehicle_Type'].value_counts().reset_index()
    production_share.columns = ['Vehicle_Type', 'Count']
    
    # Configure hover template based on display mode
    if display_mode == 'percentage':
        hover_template = '<b>%{label}</b><br>' + 'Share: %{percent}<br>' + 'Vehicles: %{value}<extra></extra>'
        textinfo = 'percent+label'
    else:  # absolute
        hover_template = '<b>%{label}</b><br>' + 'Vehicles: %{value}<br>' + 'Share: %{percent}<extra></extra>'
        textinfo = 'value+label'
    
    fig.add_trace(go.Pie(
        labels=production_share['Vehicle_Type'],
        values=production_share['Count'],
        name="Production Share",
        marker_colors=[vehicle_colors.get(vt, CORPORATE_COLORS['primary']) for vt in production_share['Vehicle_Type']],
        hovertemplate=hover_template,
        textinfo=textinfo,
        showlegend=False
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.05, 
            xanchor="center", 
            x=0.5,
            font=dict(size=9)
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family=FONT_FAMILY, color=CORPORATE_COLORS['dark']),
        margin=dict(t=100, b=120, l=60, r=60),
        barmode='stack'
    )
    
    fig.update_xaxes(title_text="Production Date", row=1, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="Vehicles Produced - Log Scale", row=1, col=1, title_font=dict(size=10), type='log')
    fig.update_xaxes(title_text="Production Date", row=1, col=2, title_font=dict(size=10))
    fig.update_yaxes(title_text="Avg Lead Time (days) - Log Scale", row=1, col=2, title_font=dict(size=10), type='log')
    fig.update_xaxes(title_text="Month", row=2, col=1, title_font=dict(size=10))
    fig.update_yaxes(title_text="Vehicles Produced", row=2, col=1, title_font=dict(size=10))
    
    return fig

# Production Pattern Summary Callback
@app.callback(
    Output('production-pattern-summary', 'children'),
    [Input('vehicle-type-dropdown', 'value'),
     Input('leadtime-range-slider', 'value')]
)
def update_production_pattern_summary(selected_types, leadtime_range):
    filtered_df = df[
        (df['Vehicle_Type'].isin(selected_types)) &
        (df['New_Lead_Time_Days'] >= leadtime_range[0]) &
        (df['New_Lead_Time_Days'] <= leadtime_range[1])
    ].copy()
    
    # Add month extraction
    filtered_df['Production_Month'] = pd.to_datetime(filtered_df['Vehicle_Prod_Date']).dt.month
    
    # Calculate seasonal patterns
    seasonal_data = filtered_df.groupby('Production_Month').agg({
        'New_Lead_Time_Days': 'mean',
        'Vehicle_Type': 'count'
    }).round(2)
    
    # Find peak and low production months
    peak_production_month = seasonal_data['Vehicle_Type'].idxmax()
    low_production_month = seasonal_data['Vehicle_Type'].idxmin()
    
    # Find most/least efficient production months
    most_efficient_month = seasonal_data['New_Lead_Time_Days'].idxmin()
    least_efficient_month = seasonal_data['New_Lead_Time_Days'].idxmax()
    
    # Calculate production share
    production_share = filtered_df['Vehicle_Type'].value_counts()
    dominant_type = production_share.index[0]
    dominant_percentage = (production_share.iloc[0] / production_share.sum() * 100)
    
    month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December']
    
    summary_text = [
        html.H4("Production Pattern Analysis Summary", style={'color': CORPORATE_COLORS['dark'], 'marginBottom': '10px', 'fontSize': '1.1rem', 'fontWeight': '600'}),
        html.P([
            f"Production portfolio analysis indicates that ", html.Strong(f"{dominant_type}"), f" dominates manufacturing volume, representing {dominant_percentage:.1f}% of total production. ",
            f"The Monthly Production Volume and Production Efficiency Trends charts utilize logarithmic scaling to better demonstrate data variations and improve visualization clarity across different value ranges."
        ], style={'color': CORPORATE_COLORS['dark'], 'fontSize': '1rem', 'lineHeight': '1.6', 'margin': '0'})
    ]
    
    return summary_text

# Factory Lead Time Distribution Callback
@app.callback(
    Output('factory-leadtime-distribution-chart', 'figure'),
    [Input('vehicle-type-dropdown', 'value'),
     Input('leadtime-range-slider', 'value')]
)
def update_factory_leadtime_distribution(selected_types, leadtime_range):
    filtered_df = df[
        (df['Vehicle_Type'].isin(selected_types)) &
        (df['New_Lead_Time_Days'] >= leadtime_range[0]) &
        (df['New_Lead_Time_Days'] <= leadtime_range[1])
    ].copy()

    # Define mappings and enforce plotting order
    component_werks = {
        'Body': 'Karosserie_Werks',
        'Engine': 'Motor_Werks',
        'Transmission': 'Schaltung_Werks',
        'Seats': 'Sitze_Werks'
    }
    component_leadtime = {
        'Body': 'Karosserie_LeadTime',
        'Engine': 'Motor_LeadTime',
        'Transmission': 'Schaltung_LeadTime',
        'Seats': 'Sitze_LeadTime'
    }
    components_order = ['Body', 'Engine', 'Transmission', 'Seats']

    # Use consistent component colors as elsewhere in the app
    component_colors = {
        'Body': CORPORATE_COLORS['primary'],
        'Transmission': CORPORATE_COLORS['secondary'],
        'Seats': CORPORATE_COLORS['accent'],
        'Engine': CORPORATE_COLORS['warning']
    }

    import plotly.graph_objs as go
    data = []
    found_data = False
    debug_text = []
    for comp in components_order:
        werks_col = component_werks.get(comp)
        lt_col = component_leadtime.get(comp)
        if werks_col is None or lt_col is None:
            continue
        if werks_col in filtered_df.columns and lt_col in filtered_df.columns:
            filtered_df[werks_col] = filtered_df[werks_col].astype(float).astype(int).astype(str)
            valid = filtered_df[werks_col].notnull() & (filtered_df[werks_col].str.strip() != "") & (filtered_df[lt_col].notnull())
            group_df = filtered_df[valid].copy()
            unique_plants = group_df[werks_col].unique().tolist()
            n_valid = len(group_df)
            debug_text.append(f"{comp}: {n_valid} valid records, unique plants: {unique_plants}")
            if not group_df.empty:
                group = group_df.groupby(werks_col)[lt_col].agg(['mean', 'count', 'max']).reset_index()
                group = group[group['count'] >= 1]
                if not group.empty:
                    found_data = True
                    group[werks_col] = group[werks_col].astype(str)
                    group['manufacturer'] = group[werks_col].str.slice(0, 3)
                    x_vals = group[werks_col].tolist()
                    customdata = group[['manufacturer', 'count']].values.tolist()
                    data.append(go.Bar(
                        x=x_vals,
                        y=group['mean'],
                        name=comp,
                        marker_color=component_colors.get(comp, CORPORATE_COLORS['primary']),
                        customdata=customdata,
                        hovertemplate=('Plant: %{x}<br>'
                                       'Avg Lead Time: %{y:.1f} days<br>'
                                       'Manufacturer: %{customdata[0]}<br>'
                                       'Samples: %{customdata[1]}')
                    ))

    fig = go.Figure(data=data)
    fig.update_layout(
        barmode='group',
        title="Factory Lead Time Distribution by Component",
        xaxis_title="Plant (first 3 digits = Manufacturer)",
        yaxis_title="Average Lead Time (days)",
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family=FONT_FAMILY, color=CORPORATE_COLORS['dark']),
        margin=dict(t=80, b=100, l=60, r=60)
    )
    if not found_data:
        debug_msg = " | ".join(debug_text)
        fig.add_annotation(text=f"No sufficient data to display factory lead time distribution.<br>{debug_msg}",
                           xref="paper", yref="paper", showarrow=False, font=dict(size=14, color=CORPORATE_COLORS['dark']),
                           x=0.5, y=0.5)
    return fig

# Factory Lead Time Summary Callback
@app.callback(
    Output('factory-leadtime-summary', 'children'),
    [Input('vehicle-type-dropdown', 'value'),
     Input('leadtime-range-slider', 'value')]
)
def update_factory_leadtime_summary(selected_types, leadtime_range):
    filtered_df = df[
        (df['Vehicle_Type'].isin(selected_types)) &
        (df['New_Lead_Time_Days'] >= leadtime_range[0]) &
        (df['New_Lead_Time_Days'] <= leadtime_range[1])
    ].copy()
    # Ensure consistent ordering and mappings
    component_werks = {
        'Body': 'Karosserie_Werks',
        'Engine': 'Motor_Werks',
        'Transmission': 'Schaltung_Werks',
        'Seats': 'Sitze_Werks'
    }
    component_leadtime = {
        'Body': 'Karosserie_LeadTime',
        'Engine': 'Motor_LeadTime',
        'Transmission': 'Schaltung_LeadTime',
        'Seats': 'Sitze_LeadTime'
    }

    components_order = ['Body', 'Engine', 'Transmission', 'Seats']

    paragraph_items = []
    min_count_for_summary = 3

    for comp in components_order:
        werks_col = component_werks.get(comp)
        lt_col = component_leadtime.get(comp)
        if werks_col in filtered_df.columns and lt_col in filtered_df.columns:
            tmp = filtered_df[[werks_col, lt_col]].copy()
            # Convert to int first to remove decimals, then to string, strip whitespace and drop empty entries
            tmp[werks_col] = tmp[werks_col].astype(float).astype(int).astype(str).str.strip()
            tmp = tmp[tmp[werks_col] != '']
            tmp = tmp[tmp[lt_col].notnull()]

            if tmp.empty:
                paragraph_items.append({'component': comp, 'text': None, 'note': 'no_data'})
                continue

            group = tmp.groupby(werks_col)[lt_col].agg(['mean', 'count']).reset_index()
            # Require a minimum number of samples per plant for a stable summary
            group = group[group['count'] >= min_count_for_summary]
            if group.empty:
                paragraph_items.append({'component': comp, 'text': None, 'note': 'insufficient'})
                continue

            # Plant with the highest average lead time
            max_idx = group['mean'].idxmax()
            max_row = group.loc[max_idx]
            plant_code = str(max_row[werks_col])
            # Robustly extract manufacturer as first 3 characters (if available)
            manufacturer_code = plant_code[:3]
            mean_lt = max_row['mean']
            count = int(max_row['count'])

            paragraph_items.append({
                'component': comp,
                'text': f"For {comp}, plant {plant_code} (manufacturer {manufacturer_code}) has the highest average lead time: {mean_lt:.1f} days (n={count}).",
                'note': 'ok'
            })
        else:
            paragraph_items.append({'component': comp, 'text': None, 'note': 'no_column'})

    # Build styled summary with an H4 title to match other summary sections
    summary_children = [
        html.H4("Factory Lead Time Distribution Summary",
                style={'color': CORPORATE_COLORS['dark'], 'marginBottom': '10px', 'fontSize': '1.1rem', 'fontWeight': '600'})
    ]

    any_ok = any(item.get('note') == 'ok' for item in paragraph_items)
    if not any_ok:
        summary_children.append(html.P("No sufficient data to display factory lead time distribution.",
                                     style={'color': CORPORATE_COLORS['dark'], 'fontSize': '1rem', 'lineHeight': '1.6', 'margin': '0'}))
        return summary_children

    # Add one paragraph per component with consistent styling
    for item in paragraph_items:
        comp = item['component']
        if item['note'] == 'ok':
            summary_children.append(html.P([
                html.Strong(f"For {comp}, "),
                item['text'].split(f"For {comp}, ", 1)[1]
            ], style={'color': CORPORATE_COLORS['dark'], 'fontSize': '1rem', 'lineHeight': '1.6', 'margin': '0'}))
        elif item['note'] == 'insufficient':
            summary_children.append(html.P([
                html.Strong(f"For {comp}, "),
                "insufficient plant-level data (fewer than 3 samples per plant) to determine a stable longest lead time."
            ], style={'color': CORPORATE_COLORS['dark'], 'fontSize': '1rem', 'lineHeight': '1.6', 'margin': '0'}))
        else:
            summary_children.append(html.P([
                html.Strong(f"For {comp}, "),
                "no component or plant data available for the selected filters."
            ], style={'color': CORPORATE_COLORS['dark'], 'fontSize': '1rem', 'lineHeight': '1.6', 'margin': '0'}))

    return summary_children

# High Lead Time Vehicles Table Callback
@app.callback(
    [Output('high-leadtime-vehicles-table', 'data'),
     Output('high-leadtime-vehicles-table', 'columns')],
    [Input('vehicle-type-dropdown', 'value'),
     Input('leadtime-range-slider', 'value')]
)
def update_high_leadtime_vehicles_table(selected_types, leadtime_range):
    filtered_df = df[
        (df['Vehicle_Type'].isin(selected_types)) &
        (df['New_Lead_Time_Days'] >= leadtime_range[0]) &
        (df['New_Lead_Time_Days'] <= leadtime_range[1])
    ].copy()
    
    # Calculate total lead time for each vehicle (sum of all component lead times)
    filtered_df['Total_Lead_Time'] = (
        filtered_df['Karosserie_LeadTime'] +
        filtered_df['Motor_LeadTime'] +
        filtered_df['Schaltung_LeadTime'] +
        filtered_df['Sitze_LeadTime']
    )
    
    # Select relevant columns for the table
    table_columns = [
        'ID_Fahrzeug',
        'Vehicle_Type', 
        'Total_Lead_Time',
        'Karosserie_LeadTime',
        'Motor_LeadTime', 
        'Schaltung_LeadTime',
        'Sitze_LeadTime'
    ]
    
    # Get top 100 vehicles with highest total lead time
    top_vehicles = filtered_df.nlargest(100, 'Total_Lead_Time')[table_columns].round(1)
    
    table_data = top_vehicles.to_dict('records')
    
    # Define column configuration with better names
    columns = [
        {'name': 'Vehicle ID', 'id': 'ID_Fahrzeug', 'type': 'text'},
        {'name': 'Type', 'id': 'Vehicle_Type', 'type': 'text'},
        {'name': 'Total Lead Time', 'id': 'Total_Lead_Time', 'type': 'numeric', 'format': {'specifier': '.1f'}},
        {'name': 'Body Lead Time', 'id': 'Karosserie_LeadTime', 'type': 'numeric', 'format': {'specifier': '.1f'}},
        {'name': 'Engine Lead Time', 'id': 'Motor_LeadTime', 'type': 'numeric', 'format': {'specifier': '.1f'}},
        {'name': 'Transmission Lead Time', 'id': 'Schaltung_LeadTime', 'type': 'numeric', 'format': {'specifier': '.1f'}},
        {'name': 'Seats Lead Time', 'id': 'Sitze_LeadTime', 'type': 'numeric', 'format': {'specifier': '.1f'}}
    ]
    
    return table_data, columns

def open_browser():
    """Open the browser automatically after a short delay."""
    webbrowser.open_new("http://127.0.0.1:8050/")

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=False, host='127.0.0.1', port=8050)
