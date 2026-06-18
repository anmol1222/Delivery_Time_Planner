import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import queue

# COnfiguration & UI Design
st.set_page_config(page_title='Smart Logistics Planner', page_icon="🚚" , layout='wide')

st.title("🚚 The Smart Logistics & Delivery Fleet Planner")
st.markdown('### AI Searching Technique (Greedy Search) + Machine Learning Heuristic')
st.write('Find the fastest path between zones by letting AI search for the route with the least predicted delay.')

#optimize cached loading
@st.cache_resource
def load_saved_model():
    if not (os.path.exists('rand_model.pkl') and os.path.exists('x_columns.pkl')):
        return None, None
    try:
        model = joblib.load('rand_model.pkl')
        columns = joblib.load('x_columns.pkl')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None
    return model, columns

model,trained_columns=load_saved_model()    

if model is None:
    st.error("⚠️ **Files Missing:** Please run your training script first to generate `rand_model.pkl` and `x_columns.pkl`!")
    
    st.stop()

ZONE_CONNECTION = {
    'Zone_1': ['Zone_2', 'Zone_3', 'Zone_4'],
    'Zone_2': ['Zone_1', 'Zone_5', 'Zone_6'],
    'Zone_3': ['Zone_1', 'Zone_6', 'Zone_7'],
    'Zone_4': ['Zone_1', 'Zone_7', 'Zone_8'],
    'Zone_5': ['Zone_2', 'Zone_9'],
    'Zone_6': ['Zone_2', 'Zone_3', 'Zone_9', 'Zone_10'],
    'Zone_7': ['Zone_3', 'Zone_4', 'Zone_10'],
    'Zone_8': ['Zone_4', 'Zone_10'],
    'Zone_9': ['Zone_5', 'Zone_6'],
    'Zone_10': ['Zone_6', 'Zone_7', 'Zone_8']
}  

def get_predicted_delay(zone, hour, day, weather):
    row= pd.DataFrame([{
        'Zone_ID': zone, 'Hour_of_Day': hour, 'Day_of_Week': day, 'Weather_Condition': weather
    }])  
    
    encoded_row = pd.get_dummies(row, columns=['Zone_ID', 'Weather_Condition'])
    final_features = encoded_row.reindex(columns=trained_columns, fill_value=0)
    
    return float(model.predict(final_features)[0])

# --- 5. AI SEARCHING TECHNIQUE (Greedy Best-First Search) ---
def find_optimal_ai_route(start, destination, hour, day, weather):
    """
    AI Search technique that navigates through connected zones.
    It acts like an intelligent GPS, querying the ML model at each step 
    to choose the neighbor with the minimum traffic delay.
    """
    visited = set()
    priority_queue = queue.PriorityQueue()
    
    # Queue structure: (priority_cost_from_ml, current_zone, current_path, accumulated_delay)
    start_delay = get_predicted_delay(start, hour, day, weather)
    priority_queue.put((start_delay, start, [start], start_delay))
    
    while not priority_queue.empty():
        _, current_zone, path, total_delay = priority_queue.get()
        
        # Target Reached! Return the route and total calculation
        if current_zone == destination:
            return path, total_delay
            
        if current_zone not in visited:
            visited.add(current_zone)
            
            # AI checks all adjacent zones connected to the current hub
            for neighbor in ZONE_CONNECTION.get(current_zone, []):
                if neighbor not in visited:
                    # ML model serves as the heuristic cost for this path choice
                    neighbor_delay = get_predicted_delay(neighbor, hour, day, weather)
                    
                    new_path = list(path) + [neighbor]
                    new_total_delay = total_delay + neighbor_delay
                    
                    # Insert into queue: Lowest traffic delay zones get prioritized first
                    priority_queue.put((neighbor_delay, neighbor, new_path, new_total_delay))
                    
    return None, 0

# --- 6. USER SIDEBAR CONTROLS ---
st.sidebar.header("🕹️ Dispatch Dashboard Inputs")

# Dynamically sort zone list for the selection menus
zone_list = sorted(list(ZONE_CONNECTION.keys()), key=lambda x: int(x.split('_')[1]))
origin = st.sidebar.selectbox("🛫 Select Origin Hub:", zone_list, index=0)
destination = st.sidebar.selectbox("🛬 Select Final Destination:", zone_list, index=9)

dispatch_hour = st.sidebar.slider("⏰ Departure Hour (24h Format):", 0, 23, 12)

day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
selected_day = st.sidebar.selectbox("📅 Weekday Profile:", list(day_map.keys()))
day_index = day_map[selected_day]

weather_forecast = st.sidebar.selectbox("🌦️ Expected Weather:", ['Clear', 'Rainy', 'Overcast', 'Foggy'])

# --- 7. EXECUTIVE CALCULATION VIEW ---
if origin == destination:
    st.warning("Please choose different zones for your Origin and Destination hubs.")
else:
    # Trigger the AI Search algorithm using real-time ML delays
    with st.spinner("AI Engine calculating optimal dispatch routing pipeline..."):
        ai_route, absolute_delay = find_optimal_ai_route(
            origin, destination, dispatch_hour, day_index, weather_forecast
        )
        
    if not ai_route:
        st.error("❌ No valid route found for the selected origin and destination.")
    else:
        # Layout Output Cards
        kpi_1, kpi_2 = st.columns(2)
        with kpi_1:
            st.metric(label="⏱️ Total Cumulative Route Delay", value=f"{absolute_delay:.1f} Minutes")
        with kpi_2:
            st.metric(label="🗺️ Total Transit Points (Hops)", value=f"{len(ai_route)} Zones")
            
        # Visualizing the final AI Path Manifest
        st.subheader("📍 AI Selected Clear Path Manifest")
        route_arrow_format = " ➡️ ".join([f"**{node}**" for node in ai_route])
        st.success(f"**Recommended Route:** {route_arrow_format}")
        
        # Step-by-Step Delay Breakdown
        st.write("#### 🔍 Route Node Bottleneck Analysis")
        breakdown_list = []
        for node in ai_route:
            node_delay = get_predicted_delay(node, dispatch_hour, day_index, weather_forecast)
            breakdown_list.append({"Logistics Hub Node": node, "Predicted Traffic Delay (Mins)": round(node_delay, 2)})
            
        st.table(pd.DataFrame(breakdown_list).set_index("Logistics Hub Node"))

