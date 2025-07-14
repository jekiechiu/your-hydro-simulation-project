import io
import json
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string
import os
from scipy.ndimage import zoom # 用於地形數據擴展 (但在此版本中不使用)

app = Flask(__name__)

# --- 水文模擬核心邏輯 ---
def run_water_simulation(original_height_map_raw, dx_original, dy_original, rainfall_type_code, simulation_duration_minutes):
    """
    運行水流模擬。
    Args:
        original_height_map_raw (np.array): 原始地形高程數據 (2D NumPy 陣列)。
        dx_original (float): 原始網格在 X 方向的間距 (m)。
        dy_original (float): 原始網格在 Y 方向的間距 (m)。
        rainfall_type_code (str): 降雨類型代碼 (A, B, C, D)。
        simulation_duration_minutes (int): 模擬總時間 (分鐘)。

    Returns:
        tuple: (
            expanded_height_map_list: list[list[float]], # 擴展後的地形高程數據 (此版本為原始數據)
            simulation_snapshots_list: list[list[list[float]]], # 每個時間步的水深數據快照
            dx_expanded: float, # 擴展後每個網格單元在 X 方向的實際間距 (此版本為原始間距)
            dy_expanded: float  # 擴展後每個網格單元在 Y 方向的實際間距 (此版本為原始間距)
        )
    """
    
    # 1. 地形數據預處理：不進行地形擴展 (細化因子固定為 1.0)
    # 這將直接使用原始地形數據，避免因插值導致的性能和記憶體問題。
    effective_subdivision_factor = 1.0 

    # 直接使用原始地形數據作為擴展後的地形
    expanded_height_map = original_height_map_raw.copy().astype(float) # 確保數據類型為浮點數

    rows, cols = expanded_height_map.shape
    
    # 計算擴展後每個小網格單元在 X 和 Y 方向上的實際物理間距
    # 由於不進行細化，這裡的擴展間距就是原始間距
    dx_expanded = dx_original / effective_subdivision_factor # 等同於 dx_original
    dy_expanded = dy_original / effective_subdivision_factor # 等同於 dy_original

    # 2. 降雨量計算
    # 根據台灣雨量規範定義的 24 小時總降雨量 (單位: mm)
    rainfall_mm_per_24hr = {
        "A": 150.0,  # 大雨 (提高降雨量，可根據需求調整)
        "B": 300.0,  # 豪雨
        "C": 500.0,  # 豪大雨
        "D": 700.0   # 超大豪雨
    }
    
    # 獲取選定的 24 小時總降雨量 (mm)
    selected_rainfall_mm_per_24hr = rainfall_mm_per_24hr.get(rainfall_type_code, rainfall_mm_per_24hr["A"])
    
    # 將 mm/24hr 轉換為 m/min
    # 1 mm = 0.001 m
    # 1 天 = 24 小時 = 1440 分鐘
    # 這是每分鐘在單位物理面積 (例如 1m x 1m) 上應增加的降雨高度 (m/min)
    base_rainfall_height_per_min = (selected_rainfall_mm_per_24hr / 1000.0) / 1440.0  # 單位: m/min

    # 每個細化後的網格單元，每分鐘接收的降雨深度
    # 這裡的修正：降雨深度是均勻的，不因網格細化而改變深度值，只改變每個單元格的物理面積
    rainfall_per_grid_cell_per_min = base_rainfall_height_per_min 

    # 3. 初始化模擬狀態
    # current_state_map 儲存每個網格點的當前總高度 (地形高程 + 積水深度)
    current_state_map = expanded_height_map.copy().astype(float) 
    
    # simulation_snapshots 儲存每個時間步的「水深」數據，用於前端視覺化
    simulation_snapshots = []
    # 在 t=0 時，水深為 0
    simulation_snapshots.append(np.full((rows, cols), 0.0, dtype=float).tolist())

    # 定義鄰居關係 (四個方向)
    neighbors_dict = {
        (i, j): [
            (i + di, j + dj)
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)] 
            if 0 <= i + di < rows and 0 <= j + dj < cols
        ]
        for i in range(rows)
        for j in range(cols)
    }

    # 4. 模擬主迴圈
    for t_step in range(1, simulation_duration_minutes + 1):
        # 每個時間步，所有網格點都接收新的雨量
        # 這是每個點在該步新接收的、用於流動和累積的"水量" (單位: 深度)
        new_rain_input = np.full((rows, cols), rainfall_per_grid_cell_per_min, dtype=float)

        # 遍歷從高到低的點，進行水流分配
        # 排序是基於 "當前總高度" (地形+累積水深)
        flat_indices = np.argsort(-current_state_map, axis=None) # 降序排列
        sorted_coords = np.unravel_index(flat_indices, current_state_map.shape)

        # 用於暫存每個點在當前時間步結束時的淨流入/流出
        net_flow_map = np.zeros((rows, cols), dtype=float)

        for r_idx, c_idx in zip(*sorted_coords):
            current_total_height = current_state_map[r_idx, c_idx]
            current_water_depth = np.maximum(0, current_total_height - expanded_height_map[r_idx, c_idx])

            # 找到比當前點 "水面總高" 更低的鄰居
            lower_neighbors = []
            for nr, nc in neighbors_dict[(r_idx, c_idx)]:
                if current_state_map[nr, nc] < current_total_height:
                    lower_neighbors.append((nr, nc))

            if lower_neighbors:
                # 找到最低的鄰居們（可能有多個）
                min_neighbor_height = min(current_state_map[nr, nc] for nr, nc in lower_neighbors)
                lowest_points = [
                    (nr, nc) for nr, nc in lower_neighbors
                    if current_state_map[nr, nc] == min_neighbor_height
                ]

                # 這裡的邏輯是簡化版，只讓新降雨流動。
                # 如果有積水，且自身水面高於最低鄰居水面，則嘗試流動部分積水
                if current_water_depth > 0 or new_rain_input[r_idx, c_idx] > 0:
                    # 計算可以流動的水量 (基於高度差)
                    # 假設在一個時間步內，水可以完全流平到最低鄰居的高度
                    
                    # 簡化流動：將新降雨和部分積水均勻分配給所有較低的鄰居
                    # 確保流動的水量不會導致自身水深為負
                    
                    # 重新考慮流動邏輯：水會從高處流向低處，直到達到相同高度或流盡
                    # 我們需要計算每個點能流出多少水，以及流入多少水
                    
                    # 簡化：每個點在每個時間步，都會嘗試將其「水面總高」高於鄰居的部分，平均分配給所有較低的鄰居
                    # 這裡只考慮新降雨的流動，不考慮已積水的流動，以避免複雜性。
                    # 如果要考慮已積水的流動，需要一個單獨的「流動階段」
                    
                    # 由於我們在循環中更新 `new_rain_distribution_map`，這已經包含了流動的概念
                    # 原始邏輯：將當前點的新接收雨量分配給最低鄰居
                    
                    # 優先流動新降雨
                    flow_value_from_rain = new_rain_input[r_idx, c_idx] / len(lowest_points)
                    for nr, nc in lowest_points:
                        net_flow_map[nr, nc] += flow_value_from_rain
                    net_flow_map[r_idx, c_idx] -= new_rain_input[r_idx, c_idx] # 從當前點流出新降雨

                    # 接著考慮積水的流動
                    # 只有當前點的水面高於最低鄰居的水面時才流動
                    if current_total_height > min_neighbor_height:
                        # 計算可以流動的積水深度（直到與最低鄰居水面齊平）
                        flowable_depth = current_total_height - min_neighbor_height
                        # 實際流動的積水深度不能超過當前積水深度
                        actual_flow_depth_from_current_water = min(current_water_depth, flowable_depth)
                        
                        if actual_flow_depth_from_current_water > 0:
                            flow_value_from_current_water = actual_flow_depth_from_current_water / len(lowest_points)
                            for nr, nc in lowest_points:
                                net_flow_map[nr, nc] += flow_value_from_current_water
                            net_flow_map[r_idx, c_idx] -= actual_flow_depth_from_current_water

            else:
                # 如果沒有更低的鄰居，水就留在原地，累積新降雨
                net_flow_map[r_idx, c_idx] += new_rain_input[r_idx, c_idx]


        # 更新當前時間步的總高度：前一時間步的總高度 + 淨流入/流出
        current_state_map = current_state_map + net_flow_map
        
        # 確保總高度不會低於原始地形高度（即水深不會為負）
        current_state_map = np.maximum(expanded_height_map, current_state_map)

        # 計算當前時間步的實際水深 (總高度 - 原始地形高度)
        water_depth_at_t = np.maximum(0, current_state_map - expanded_height_map)
        simulation_snapshots.append(water_depth_at_t.tolist())
    
    return expanded_height_map.tolist(), simulation_snapshots, dx_expanded, dy_expanded


# 全局變數用於儲存地形和模擬結果
global_original_terrain = None # 擴展後的原始地形，用於前端顯示灰色網格
global_simulation_snapshots = None # 每個時間步的水深數據
global_sim_current_step = 0
global_dx_expanded = 0.0 # 儲存擴展後的 dx
global_dy_expanded = 0.0 # 儲存擴展後的 dy

# --- 路由定義 ---

# 提供前端 HTML 檔案
@app.route('/')
def serve_index():
    with open('index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return render_template_string(html_content)

# 處理 Excel 檔案上傳和模擬參數
@app.route('/upload-and-simulate', methods=['POST'])
def upload_and_simulate():
    global global_original_terrain, global_simulation_snapshots, global_sim_current_step, global_dx_expanded, global_dy_expanded

    if 'file' not in request.files:
        return {"error": "No file part"}, 400
    file = request.files['file']
    if file.filename == '':
        return {"error": "No selected file"}, 400
    
    if file and file.filename.endswith(('.xlsx', '.xls')): # 支援 .xls
        try:
            # 假設 Excel 檔案沒有標頭，直接讀取所有數據
            df = pd.read_excel(io.BytesIO(file.read()), header=None) 
            original_height_map_raw = df.to_numpy()
            
            # 從表單數據獲取模擬參數
            dx = float(request.form.get('dx', 10.0)) # 預設值 10m
            dy = float(request.form.get('dy', 10.0))
            rainfall_type = request.form.get('rainfall_type', 'A') # 預設 A
            simulation_time_min = float(request.form.get('sim_time', 60.0)) # 分鐘
            
            # 運行水流模擬
            global_original_terrain, global_simulation_snapshots, global_dx_expanded, global_dy_expanded = run_water_simulation(
                original_height_map_raw, dx, dy, rainfall_type, int(simulation_time_min)
            )
            global_sim_current_step = 0 # 重置模擬步長

            rows_expanded = len(global_original_terrain)
            cols_expanded = len(global_original_terrain[0]) if rows_expanded > 0 else 0

            return {
                "status": "success",
                "message": "Simulation started",
                "terrain_data": global_original_terrain, # 這裡返回的是擴展後的地形
                "rows": rows_expanded,
                "cols": cols_expanded,
                "dx": dx, # 原始 dx, dy (用於前端顯示)
                "dy": dy,
                "dx_expanded": global_dx_expanded, # 擴展後的實際網格間距 (用於前端 Three.js 尺寸和水量計算)
                "dy_expanded": global_dy_expanded, # 擴展後的實際網格間距 (用於前端 Three.js 尺寸和水量計算)
                "total_steps": len(global_simulation_snapshots) # 總模擬步數 (包含 t=0)
            }
        except Exception as e:
            # 捕捉讀取Excel或模擬計算中的錯誤
            app.logger.error(f"Error processing Excel or running simulation: {e}", exc_info=True)
            return {"error": f"Failed to process Excel or run simulation: {str(e)}"}, 500
    else:
        return {"error": "Invalid file type. Please upload an .xlsx or .xls file."}, 400

# 提供實時水深數據
@app.route('/get-water-data', methods=['GET'])
def get_water_data():
    global global_simulation_snapshots, global_sim_current_step, global_dx_expanded, global_dy_expanded

    if global_simulation_snapshots is None or not global_simulation_snapshots:
        return {"error": "No simulation running or no snapshots available"}, 400
    
    if global_sim_current_step < len(global_simulation_snapshots):
        water_data = global_simulation_snapshots[global_sim_current_step]
        current_step_display = global_sim_current_step # 用於顯示的步數
        global_sim_current_step += 1 # 為下次請求增加
        return {
            "water_depth": water_data,
            "current_step": current_step_display, # 返回當前讀取的步數
            "total_steps": len(global_simulation_snapshots),
            "dx_expanded": global_dx_expanded, # 每次都傳遞擴展後的實際網格間距
            "dy_expanded": global_dy_expanded  # 每次都傳遞擴展後的實際網格間距
        }
    else:
        # 如果所有快照都已發送，通知前端模擬結束
        return {"status": "simulation_finished", "message": "No more water data"}

if __name__ == '__main__':
    # 在生產環境中，應使用 Gunicorn 或其他 WSGI 服務器
    # 例如: gunicorn -w 4 app:app
    app.run(debug=True) # debug=True 僅用於開發環境
