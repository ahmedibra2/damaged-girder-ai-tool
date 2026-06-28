import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

# ----------------------------
# Load model and scaler
# ----------------------------
model = xgb.XGBRegressor()
model.load_model("xgb_model.json")

scaler = joblib.load("scaler.pkl")

# ----------------------------
# Streamlit page configuration
# ----------------------------
st.set_page_config(page_title="Damaged Girder AI Diagnostic Tool", layout="wide")

# ----------------------------
# Load LIME explainer 
# ----------------------------
@st.cache_resource
def load_lime_explainer_and_names():
    assets = joblib.load("lime_assets.pkl")
    X_train_raw_local = assets["X_train_raw"]
    feature_names_lime_local = assets["feature_names_lime"]
    cfg = assets.get("lime_config", {})

    lime_explainer = LimeTabularExplainer(
        training_data=X_train_raw_local,
        feature_names=feature_names_lime_local,
        **cfg
    )
    return lime_explainer, feature_names_lime_local

lime_explainer, feature_names_lime = load_lime_explainer_and_names() 

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("AI-based Assessment Tool for Damaged Steel Girders")

c1, c2 = st.columns([1.5, 1])

with c1:
    st.image(
        "beam_schematic.png",
        caption="Schematic of Damage Locations along Beam Span",
        use_container_width=True,
    )

with c2:
    st.image("value_ranges.png", caption="Range of Girder Properties Used in Training",
        # use_container_width=True,
        width=300,
    )

st.markdown("#### Girder Section Properties")

col1, col2, col3 = st.columns(3)
with col2:
    st.image("girder_section.png", caption="Girder Cross-sectional Parameters", width=250)

# input mode
mode = st.radio("Input Mode", ["Manual entry", "Select AISC section"], horizontal=True)

L = d = bf = tf = tw = hw = Ix = Iy = rx = ry = 0.0

if mode == "Manual entry":

    st.write("Enter the following girder section properties:")

    col1, col2 = st.columns(2)
    with col1:
        L = st.number_input(r"Span Length $L$ (in.)", value=0.0, step=50.0, format="%.0f")
        bf = st.number_input(r"Flange Width $b_f$ (in.)", value=0.0, step=0.1, format="%.2f")
        tw = st.number_input(r"Web Thickness $t_w$ (in.)", value=0.0, step=0.01, format="%.2f")
    with col2:
        d  = st.number_input(r"Girder Depth $d$ (in.)", value=0.0, step=1.0, format="%.2f")
        tf = st.number_input(r"Flange Thickness $t_f$ (in.)", value=0.0, step=0.01, format="%.2f")
        hw = st.number_input(r"Web Height $h_w$ (in.)", value=0.0, step=1.0, format="%.2f")

    st.markdown("#### Auto-computed section properties")
        
    if L > 0 and d > 0 and bf > 0 and tf > 0 and tw > 0 and hw > 0:
        Ix = (bf*d**3/12) - ((bf - tw)*(d - 2*tf)**3/12)
        Iy = ((d - 2*tf)*tw**3/12) + 2*(tf*bf**3/12)
        A = 2*bf*tf + hw*tw
        rx = np.sqrt(Ix / A)
        ry = np.sqrt(Iy / A)
        Zx = 0.25*hw**2 * tw + bf*tf * (hw + tf) 

        st.write(f"Radius of Gyration about x-axis $r_x$: {rx:.2f} in.")
        st.write(f"Radius of Gyration about y-axis $r_y$: {ry:.2f} in.")
        st.write(f"Moment of Inertia about x-axis $I_x$: {Ix:.0f} in.$^4$")
        st.write(f"Moment of Inertia about y-axis $I_y$: {Iy:.0f} in.$^4$")

else:
    aisc_df = pd.read_csv("aisc_database_truncated.csv")

    # select AISC section
    section_col = aisc_df.columns[0]
    sections = aisc_df[section_col].astype(str).tolist()
    sel = st.selectbox("Select AISC Section", sections)

    # Span is not in AISC DB, so user provides it
    L = st.number_input(r"Span Length $L$ (in.)", value=0.0, step=50.0, format="%.0f")

    # automatically fill properties
    row = aisc_df[aisc_df[section_col] == sel].iloc[0]
    d  = float(row["d"])
    bf = float(row["bf"])
    tf = float(row["tf"])
    tw = float(row["tw"])
    hw = float(row["hw"])
    rx = float(row["rx"])
    ry = float(row["ry"])
    Ix = float(row["Ix"])
    Iy = float(row["Iy"])
    Zx = float(row["Zx"])

    # Display properties
    st.markdown("#### AISC Section Properties")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Girder Depth $d$: {d:.2f} in.")
        st.write(f"Flange Width $b_f$: {bf:.2f} in.")
        st.write(f"Flange Thickness $t_f$: {tf:.3f} in.")
        st.write(f"Web Thickness $t_w$: {tw:.3f} in.")

    with c2:
        st.write(f"Web Height $h_w$: {hw:.2f} in.")
        st.write(f"Radius of Gyration about x-axis $r_x$: {rx:.2f} in.")
        st.write(f"Radius of Gyration about y-axis $r_y$: {ry:.2f} in.")
        st.write(f"Moment of Inertia about x-axis $I_x$: {Ix:.0f} in.$^4$")
        st.write(f"Moment of Inertia about y-axis $I_y$: {Iy:.0f} in.$^4$")

# Damage Inputs
st.markdown("#### Damage Measurements")

col1, col2, col3 = st.columns(3)
with col2:
    st.image("girder_damage.png", caption="Schematic Reference for Damage Measurements", width=300)

Dx = st.number_input(r"Horizontal Damage Measurement $D_x$ (in.)", value=0.0, step=0.5, format="%.2f")
Dy = st.number_input(r"Vertical Damage Measurement $D_y$ (in.)", value=0.0, step=0.5, format="%.2f")
# DLR = st.number_input(r"Damage Location Ratio to Span $DLR$ (dimensionless)", value=0.0, step=0.25, format="%.2f")
# DLR = st.selectbox(
#     r"Damage Location Ratio to Span $DLR$ (dimensionless)",
#     options=[None, 0.25, 0.50, 0.75, 0.85],
#     index=0,
#     format_func=lambda x: "Select..." if x is None else f"{x:.2f}",
# )
# DLR = st.radio(r"Damage Location Ratio to Span $DLR$ (dimensionless)", [0.25, 0.50, 0.75, 0.85], horizontal=True)

# Damage location input
damage_location = st.number_input(
    "Damage Location from support (in.)",
    min_value=0.0,
    value=0.0,
    step=50.0,
)

reference_support = st.radio(
    "Measured from which support:",
    ["Exterior Support", "Interior Support"],
    horizontal=True,
)

# Compute DLR
if L > 0:
    if reference_support == "Exterior Support":
        DLR = damage_location / L
    else:  # Interior Support
        DLR = 1.0 - (damage_location / L)

    st.write(f"Computed Damage Location Ratio (DLR) = {DLR:.2f}")

# ----------------------------
# Function to validate all inputs have been provided
# ----------------------------
def validate_inputs(values, mins):
    missing = []
    for label, v in values.items():
        if v is None:
            missing.append(label)
            continue

        min_allowed = mins.get(label, 0.0)  # default: require > 0
        if float(v) <= min_allowed:            # inclusive min allowed
            missing.append(label)

    return missing

# ----------------------------
# Function to check ranges of inputs against training data ranges
# ----------------------------
def check_training_ranges(values, ranges):
    out_of_range = []

    for label, v in values.items():
        if v is None:
            continue

        min_val, max_val = ranges[label]

        if float(v) < min_val or float(v) > max_val:
            out_of_range.append((label, float(v), min_val, max_val))

    return out_of_range

if st.button("Run Prediction"):

    required = {
        r"Span Length $L$": L,
        r"Girder Depth $d$": d,
        r"Flange Width $b_f$": bf,
        r"Flange Thickness $t_f$": tf,
        r"Web Thickness $t_w$": tw,
        r"Web Height $h_w$": hw,
        r"Damage Location Ratio $DLR$": DLR,
        r"Horizontal Damage $D_x$": Dx,
        r"Vertical Damage $D_y$": Dy,
    }

    mins = {
        r"Span Length $L$": 0.0,
        r"Girder Depth $d$": 0.0,
        r"Flange Width $b_f$": 0.0,
        r"Flange Thickness $t_f$": 0.0,
        r"Web Thickness $t_w$": 0.0,
        r"Web Height $h_w$": 0.0,
        r"Damage Location Ratio $DLR$": 0.0,   
        r"Horizontal Damage $D_x$": 1.0,       
        r"Vertical Damage $D_y$": 1.0,        
    }

    missing = validate_inputs(required, mins)

    training_ranges = {
                        r"Span Length $L$": (450, 1050),
                        r"Girder Depth $d$": (24.58, 36.84),
                        r"Flange Width $b_f$": (8.99, 15.0),
                        r"Flange Thickness $t_f$": (0.64, 1.19),
                        r"Web Thickness $t_w$": (0.44, 0.71),
                        r"Web Height $h_w$": (23.22, 34.96),
                        r"Radius of Gyration $r_x$": (9.91, 14.62),
                        r"Radius of Gyration $r_y$": (1.92, 3.44),
                        r"Moment of Inertia $I_x$": (2205, 9936),
                        r"Moment of Inertia $I_y$": (82.51, 670.3),
                        r"Horizontal Damage $D_x$": (0.0, 19.79),
                        r"Vertical Damage $D_y$": (0.0, 10.91),
                        r"Damage Location Ratio $DLR$": (0.25, 0.85),
                    }
    
    input_values = {
                    r"Span Length $L$": L,
                    r"Girder Depth $d$": d,
                    r"Flange Width $b_f$": bf,
                    r"Flange Thickness $t_f$": tf,
                    r"Web Thickness $t_w$": tw,
                    r"Web Height $h_w$": hw,
                    r"Radius of Gyration $r_x$": rx,
                    r"Radius of Gyration $r_y$": ry,
                    r"Moment of Inertia $I_x$": Ix,
                    r"Moment of Inertia $I_y$": Iy,
                    r"Horizontal Damage $D_x$": Dx,
                    r"Vertical Damage $D_y$": Dy,
                    r"Damage Location Ratio $DLR$": DLR,
                }

    out_of_range = check_training_ranges(input_values, training_ranges)

    if out_of_range:
        st.warning(
            "One or more inputs are outside the range of the training dataset. "
            "The prediction may be less reliable."
        )

        with st.expander("Show out-of-range variables"):
            for label, value, min_val, max_val in out_of_range:
                st.write(
                    f"* {label} value is {value:.3g}; "
                    f"training range is between {min_val:.3g} and {max_val:.3g}."
            )

    if DLR <= 0 or DLR >= 1:
        missing.append("DLR must be between 0 and 1")

    if missing:
        st.error("Please complete the required inputs before running the prediction.")
        st.markdown("**Missing or invalid fields:**")
        st.write("\n".join([f"* {m}" for m in missing]))
        st.stop()
    # ----------------------------
    
    X_raw = np.array([[L, d, bf, tf, tw, hw, rx, ry, Ix, Iy, Dx, Dy, DLR]])

    # Scale inputs for model
    X_scaled = scaler.transform(X_raw)

    # Run prediction
    C_R = model.predict(X_scaled)[0]
    L_ft = L / 12 # ft
    w_R = C_R / (2*L_ft) # kips/ft

    # Compute Residual Capacity Ratio (two-span 36 ksi girder)
    Fy = 36 # ksi
    C = 24 * Fy * Zx / L # full capacity in kips
    residual_ratio = C_R / C

    # ----------------------------
    # AASHTOWare Inputs
    # ----------------------------
    a = L_ft * DLR # ft
    Mp = (Fy*Zx) / 12 # kips-ft
    

    M_red = a * ((-Mp / L_ft) + (w_R * L_ft/2) - (w_R * a/2))
    penalty_percent = (1 - residual_ratio) * 100

    # -----------------------------
    # Display results and intermediate calculations
    # -----------------------------
    st.divider()
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Results")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        with st.container(border=True):
            penalty_display = f"{penalty_percent:.1f} %" if penalty_percent > 0 else "N/A"
            st.metric("Additional Self-load %", penalty_display)
    with c2:
        with st.container(border=True):
            st.metric("Girder uniform load capacity", f"{w_R:.2f} kips/ft")

    # Intermediate calculations
    with st.expander("Show intermediate steps"):
        st.divider()

        st.write(fr"Yield Stress $F_y = {Fy} \,\, \text{{ksi}}$")
        st.write(fr"Plastic Section Modulus $Z_x = {Zx:.0f} \,\, \text{{in}}^4$")
        st.write(f"**Equivalent Reduced Uniform Load Calculation:**")

        st.latex(fr"w_R = \frac{{C_R}}{{2L}} = \frac{{{C_R:.1f}}}{{2 * {L_ft:.1f}}} = {w_R:.2f} \,\, \text{{kips/ft}}")
        st.write("Where")
        st.write(r"$C_R$: Residual Force from AI Model (kips)")
        st.write(r"$L$: Span Length (ft)")

        st.divider()

        st.write(f"**Penalty Percentage Calculation:**")
        st.latex(r"\text{Penalty (\%)} = (1 - \frac{C_R}{C}) \times 100")
        if penalty_percent > 0:
            st.latex(fr"\text{{Penalty (\%)}} = (1 - \frac{{{C_R:.1f}}}{{{C:.1f}}}) \times 100 = {penalty_percent:.1f} \%")
        st.write("Where")
        st.write(r"$C_R$: Residual Force from AI Model (kips)")
        st.write(fr"$C$: Undamaged Capacity (kips)  $\therefore$  $\frac{{24 \, Fy \, Zx}}{{L}} = \frac{{24 * 36 * {Zx:.0f}}}{{{L:.0f}}} = {C:.1f} \,\, \text{{kips}}$")

    # ----------------------------
    # LIME Explainer Plots
    # ----------------------------
    st.divider()
    left, right = st.columns([0.9, 1.5])

    def predict_fn(X: np.ndarray) -> np.ndarray:
        X_scaled_local = scaler.transform(X)
        return model.predict(X_scaled_local)

    x_instance = X_raw.flatten()

    with st.spinner("Computing parameter explanation..."):
        exp_lime = lime_explainer.explain_instance(
            data_row=x_instance,
            predict_fn=predict_fn,
            num_features=13
        )

    with left:
        st.markdown("#### **Parameters Influence on Predicted Capacity**")
        st.markdown("**Note:** Red contributions indicate parameters that decrease the predicted capacity.")
        fig_lime = exp_lime.as_pyplot_figure()
        fig_lime.tight_layout()
        st.pyplot(fig_lime, use_container_width=True)
        plt.close(fig_lime)

#----------------------------
# Disclaimer
#----------------------------
st.divider()
with st.expander("Copyright & Disclaimer Notice", expanded=False):
    st.markdown(
                """
        Copyright © 2026 ICT R27-255 Research Team at the University of Illinois Urbana-Champaign. All rights reserved.

        This software has been developed based on two-span steel bridge girder data with A36 steel grade only. 
        Accuracy of results generated outside the specified ranges of girder properties might require further validation.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.

        Report bugs to ahmedei2@illinois.edu.
                """
            )
