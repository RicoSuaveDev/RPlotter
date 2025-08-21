import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from matplotlib.colors import to_rgb

# Enable interactive mode for matplotlib
plt.ion()

def darken_color(color, factor=0.6):
    """Darken a color by multiplying RGB values by factor"""
    rgb = to_rgb(color)
    return tuple(val * factor for val in rgb)

# Define color schemes
COLOR_SCHEMES = {
    'Blue Pastels': ['#A6CAF0', '#89B9E2', '#6DA8D6', '#5098CA', '#3387BE', '#1677B2', '#0066A6'],
    'Green Pastels': ['#B5E2BE', '#98D4A4', '#7BC68A', '#5EB870', '#41AA56', '#249C3C', '#078E22'],
    'Purple Pastels': ['#E0C6E4', '#D2ABD6', '#C490C8', '#B675BA', '#A85AAC', '#9A3F9E', '#8C2490'],
    'Pink Pastels': ['#FFD1DC', '#FFB6C1', '#FF9AA2', '#FF7F83', '#FF6464', '#FF4945', '#FF2D26'],
    'Orange Pastels': ['#FFD4B2', '#FFC299', '#FFB080', '#FF9E66', '#FF8C4D', '#FF7A33', '#FF681A'],
    'Rainbow': ['#FF9AA2', '#98FB98', '#87CEFA', '#FFD700', '#FFA07A', '#DDA0DD', '#98FB98'],  # More vibrant rainbow colors
    'Neon': ['#FF1177', '#00FF00', '#00FFFF', '#FF00FF', '#FFFF00', '#1FFF1F', '#FF4444'],  # Bright neon colors
    'Candy': ['#FF0090', '#FF3F3F', '#00B2FF', '#9F00FF', '#00FF00', '#FFB700', '#FF60A6'],  # Sweet vibrant colors
    'Electric': ['#00FF00', '#FF00FF', '#00FFFF', '#FF3300', '#3300FF', '#FFFF00', '#FF0099'],  # Bold electric colors
    'Sunset': ['#FF0000', '#FF4D00', '#FF9900', '#FFCC00', '#FFFF00', '#FF00CC', '#FF33FF']  # Bright warm colors
}

def get_colors(scheme_name, num_items):
    """Get colors from a scheme, interpolating if needed"""
    base_colors = COLOR_SCHEMES[scheme_name]
    if num_items <= len(base_colors):
        return base_colors[:num_items]
    else:
        # Interpolate colors if we need more than we have
        from matplotlib.colors import LinearSegmentedColormap, to_rgb
        base_rgb_colors = [to_rgb(c) for c in base_colors]
        cmap = LinearSegmentedColormap.from_list("custom", base_rgb_colors)
        return [plt.matplotlib.colors.rgb2hex(cmap(i)) for i in np.linspace(0, 1, num_items)]

st.set_page_config(page_title="CSV Data Plotter", page_icon="📊", layout="wide")

# Create two main columns for the layout
left_col, right_col = st.columns([1, 1], gap="large")

# Initialize df as None
df = None

with left_col:
    st.title("📊 CSV Data Plotter")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("### Data Preview")
        st.dataframe(df.head(50), use_container_width=True)

# Only proceed with the rest of the app if we have data
if df is not None:

    # Move chart controls and preview to right column
    with right_col:
        st.header("Chart Preview")
        
        # Chart controls in a container
        with st.container():
            st.subheader("Chart Controls")
            if df is None:
                st.info("Please upload a CSV file to begin")
            
            cols = list(df.columns)
            x_col = st.selectbox("X-axis column", cols, index=0)
            
            # Get all numerical columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numeric_cols.append("Count")  # Add Count as an option
            
            # Y-axis selection
            y_col = st.selectbox("Y-axis column", numeric_cols, index=len(numeric_cols)-1)  # Set Count as default

            # Chart configuration in two rows
            control_col1, control_col2 = st.columns(2)
            with control_col1:
                chart_type = st.selectbox("Chart type", ["Bar", "Horizontal Bar", "Line", "Area", "Pie"], index=0)
                max_items = st.select_slider(
                    "Maximum items to show",
                    options=[5, 10, 15, 20, 25, 30, 'All'],
                    value=15
                )
            with control_col2:
                sort_options = ["Value (Descending)", "Value (Ascending)", "Name (A-Z)", "Name (Z-A)"] if y_col != "Count" else ["Count (Descending)", "Count (Ascending)", "Name (A-Z)", "Name (Z-A)"]
                sort_by = st.selectbox("Sort by", sort_options, index=0)
                show_values = st.toggle("Show values on chart", value=False)
                color_scheme = st.selectbox("Color Scheme", list(COLOR_SCHEMES.keys()), index=0)
                dark_mode = st.toggle("Dark Mode", value=False)

    # Prepare data
    if y_col == "Count":
        # For Count, group by X and count occurrences
        plot_df = df.groupby(x_col).size().reset_index(name="Count")
    else:
        # For numeric columns, calculate mean, min, max, and count
        agg_dict = {
            y_col: ['mean', 'min', 'max', 'count']
        }
        plot_df = df.groupby(x_col).agg(agg_dict).reset_index()
        plot_df.columns = [x_col, f"{y_col}_mean", f"{y_col}_min", f"{y_col}_max", f"{y_col}_count"]
        
        # Add statistics to the left column
        with left_col:
            st.write(f"### Statistics for {y_col}")
            stats_cols = st.columns(4)
            with stats_cols[0]:
                st.metric("Mean", f"{plot_df[f'{y_col}_mean'].mean():.2f}")
            with stats_cols[1]:
                st.metric("Min", f"{plot_df[f'{y_col}_min'].min():.2f}")
            with stats_cols[2]:
                st.metric("Max", f"{plot_df[f'{y_col}_max'].max():.2f}")
            with stats_cols[3]:
                st.metric("Total Count", f"{plot_df[f'{y_col}_count'].sum():,}")
        
        # Use mean for plotting
        plot_df = plot_df[[x_col, f"{y_col}_mean"]].rename(columns={f"{y_col}_mean": y_col})
    
    # Sort data based on user selection
    if sort_by == "Count (Descending)" or sort_by == "Value (Descending)":
        plot_df = plot_df.sort_values(y_col, ascending=False)
    elif sort_by == "Count (Ascending)" or sort_by == "Value (Ascending)":
        plot_df = plot_df.sort_values(y_col, ascending=True)
    elif sort_by == "Name (A-Z)":
        plot_df = plot_df.sort_values(x_col)
    else:  # Name (Z-A)
        plot_df = plot_df.sort_values(x_col, ascending=False)
    
    # Limit the number of items if not 'All'
    if max_items != 'All':
        plot_df = plot_df.head(max_items)

    # Create the chart with a larger figure size for better interactivity
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.toolbar_visible = True
    fig.canvas.header_visible = True
    
    # Set dark mode styles if enabled
    if dark_mode:
        # Set dark background
        fig.patch.set_facecolor('#1E1E1E')
        ax.set_facecolor('#2D2D2D')
        
        # Set grid and spine colors
        ax.grid(True, linestyle='--', alpha=0.3, color='#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.spines['top'].set_color('#666666')
        ax.spines['left'].set_color('#666666')
        ax.spines['right'].set_color('#666666')
        
        # Set tick colors
        ax.tick_params(colors='#FFFFFF')
        
        # Set label colors
        ax.xaxis.label.set_color('#FFFFFF')
        ax.yaxis.label.set_color('#FFFFFF')
        ax.title.set_color('#FFFFFF')
    else:
        # Light mode settings
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Convert x-axis values to strings for categorical plotting
    plot_df[x_col] = plot_df[x_col].astype(str)

    # Get colors for the current plot
    colors = get_colors(color_scheme, len(plot_df))

    if chart_type == "Bar":
        # Create darker edge colors for each bar
        edge_colors = [darken_color(color) for color in colors]
        bars = ax.bar(plot_df[x_col], plot_df[y_col], color=colors, 
                     edgecolor=edge_colors, linewidth=0.8)
        if show_values:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom',
                        color='white' if dark_mode else 'black')
    elif chart_type == "Horizontal Bar":
        edge_colors = [darken_color(color) for color in colors]
        bars = ax.barh(plot_df[x_col], plot_df[y_col], color=colors,
                      edgecolor=edge_colors, linewidth=0.8)
        if show_values:
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{width:.2f}', ha='left', va='center',
                        color='white' if dark_mode else 'black')
    elif chart_type == "Line":
        # Add colored lines with darker edges
        for i in range(len(plot_df) - 1):
            ax.plot(plot_df[x_col].iloc[i:i+2], plot_df[y_col].iloc[i:i+2], 
                   color=colors[i], linewidth=2, zorder=2)
            ax.plot(plot_df[x_col].iloc[i:i+2], plot_df[y_col].iloc[i:i+2], 
                   color=darken_color(colors[i]), linewidth=0.8, zorder=3)
        # Add points with darker edge of the same color
        for i, (x, y) in enumerate(zip(plot_df[x_col], plot_df[y_col])):
            ax.scatter(x, y, color=colors[i], 
                      edgecolor=darken_color(colors[i]), 
                      linewidth=0.8, s=80, zorder=4)
        if show_values:
            for x, y in zip(plot_df[x_col], plot_df[y_col]):
                ax.text(x, y, f'{y:.2f}', ha='center', va='bottom',
                       color='white' if dark_mode else 'black')
    elif chart_type == "Area":
        for i in range(len(plot_df) - 1):
            ax.fill_between(range(i, i+2), plot_df[y_col].iloc[i:i+2], 
                          color=colors[i], alpha=0.6)
            # Add outline in darker shade of the same color
            ax.plot(range(i, i+2), plot_df[y_col].iloc[i:i+2],
                   color=darken_color(colors[i]), linewidth=0.8, zorder=3)
        # Add points with darker edge of the same color
        for i in range(len(plot_df)):
            ax.scatter(i, plot_df[y_col].iloc[i], color=colors[i],
                      edgecolor=darken_color(colors[i]), 
                      linewidth=0.8, s=80, zorder=4)
        if show_values:
            for i, y in enumerate(plot_df[y_col]):
                ax.text(i, y, f'{y:.2f}', ha='center', va='bottom',
                       color='white' if dark_mode else 'black')
        ax.set_xticks(range(len(plot_df)))
        ax.set_xticklabels(plot_df[x_col])
    elif chart_type == "Pie":
        # Create darker edge colors for each slice
        edge_colors = [darken_color(color) for color in colors]
        values = plot_df[y_col].values  # Get values as numpy array for easier indexing
        
        def make_autopct(values, show_val):
            def my_autopct(pct):
                # Find the corresponding value based on percentage
                idx = int(round(pct * len(values) / 100.0)) - 1
                if idx < 0:  # Handle edge case for small percentages
                    idx = 0
                val = values[min(idx, len(values)-1)]  # Ensure we don't exceed array bounds
                if show_val:
                    return f'{pct:.1f}%\n({val:.2f})'
                return f'{pct:.1f}%'
            return my_autopct
        
        ax.pie(plot_df[y_col], labels=plot_df[x_col].astype(str), 
               colors=colors,
               wedgeprops={'edgecolor': 'none', 'linewidth': 0.8},  # Initialize without edge
               autopct=make_autopct(values, show_values),
               textprops={'color': 'white' if dark_mode else 'black'})
        
        # Update edge colors for each wedge individually
        for wedge, edge_color in zip(ax.patches, edge_colors):
            wedge.set_edgecolor(edge_color)
    
    # Set labels and title
    if chart_type != "Horizontal Bar":
        ax.set_xlabel(x_col)
    if chart_type != "Pie":
        if chart_type == "Horizontal Bar":
            ax.set_ylabel(x_col)
            ax.set_xlabel(y_col)
        else:
            ax.set_ylabel(y_col)
    
    ax.set_title(f"{chart_type} Chart: {y_col} by {x_col}", 
                 color='white' if dark_mode else 'black')
    
    # Adjust label size and rotation
    if chart_type != "Pie":
        if chart_type == "Horizontal Bar":
            plt.yticks(fontsize=8, color='white' if dark_mode else 'black')
        else:
            plt.xticks(fontsize=8, rotation=45, ha='right', color='white' if dark_mode else 'black')
            plt.subplots_adjust(bottom=0.2)
        
        # Update tick label colors
        ax.tick_params(axis='both', colors='white' if dark_mode else 'black')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color('white' if dark_mode else 'black')
    
    # Automatically adjust subplot parameters for better layout
    plt.tight_layout()

    # Create chart container in the right column
    with right_col:
        chart_container = st.container()
        with chart_container:
            st.markdown(
                """
                <style>
                    .stImage > img {
                        max-width: 100% !important;
                        max-height: 320px !important;
                    }
                </style>
                """, 
                unsafe_allow_html=True
            )
            st.pyplot(fig, use_container_width=True)

        # ✅ Download chart as PNG - moved to left column
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        
    # Add download button to left column
    with left_col:
        st.download_button(
            "Download chart as PNG",
            buf.getvalue(),
            file_name="chart.png",
            mime="image/png"
        )
