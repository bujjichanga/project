# from flask import Flask, render_template, request
# import os
# import pandas as pd
# from prophet import Prophet
# import matplotlib.pyplot as plt

# app = Flask(__name__)
# UPLOAD_FOLDER = os.path.join('static')
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load product data
# product_data = pd.read_csv('price/preprocessed_data.csv')
# category_product_map = (
#     product_data.groupby('Category')['Product']
#     .unique().apply(list).to_dict()
# )

# @app.route('/')
# def homepage():
#     return render_template('homepage.html')

# @app.route('/forecast', methods=['GET', 'POST'])
# def forecast():
#     if request.method == 'POST':
#         category = request.form['category']
#         product = request.form['product']

#         df = pd.read_csv("price/preprocessed_data.csv")
#         df = df[(df['Category'] == category) & (df['Product'] == product)]

#         if df.empty:
#             return render_template("index.html",
#                                    categories=list(category_product_map.keys()),
#                                    product_map=category_product_map,
#                                    error="No data available for selected category and product.")

#         df = df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})
#         df['ds'] = pd.to_datetime(df['ds'])

#         model = Prophet()
#         model.fit(df)
#         future = model.make_future_dataframe(periods=30)
#         forecast = model.predict(future)

#         avg_price = round(forecast['yhat'].mean(), 2)
#         peak_price = round(forecast['yhat'].max(), 2)

#         # Determine price trend direction
#         trend_start = forecast['yhat'].iloc[-31]
#         trend_end = forecast['yhat'].iloc[-1]
#         is_trend_decreasing = trend_end < trend_start

#         trend_message = "🔻 The product price trend is decreasing." if is_trend_decreasing else "🔺 The product price trend is increasing."

#         # Generate insights based on trend direction
#         if is_trend_decreasing:
#             buyer_insights = [
#                 "Better Deals Ahead: Prices are expected to drop, so they might wait to purchase later at a lower cost.",
#                 "Smart Budgeting: Helps plan purchases for essentials or high-priced items when prices become more affordable.",
#                 "Avoid Overpaying: Reduces the chance of buying at a high price when a dip is predicted soon."
#             ]
#             seller_insights = [
#                 "Adjust Pricing Strategy: Sellers might need to lower prices to stay competitive as market prices fall.",
#                 "Inventory Planning: If prices are falling, sellers might want to clear stock faster before it loses more value.",
#                 "Promotional Timing: They could run offers early or bundle products to maintain profit margins before the drop."
#             ]
#         else:
#             buyer_insights = [
#                 "Buy Now Before Prices Soar: If prices are climbing, purchasing early could save money.",
#                 "Prioritize Essential Items: Get important products first before costs go higher.",
#                 "Look for Early Deals: Seek discounts or offers before prices increase more."
#             ]
#             seller_insights = [
#                 "Higher Profits Potential: Sellers may benefit from upcoming higher price points.",
#                 "Hold Stock Strategically: Consider timing sales to align with peak value.",
#                 "Launch at Higher Price Points: Ideal time to introduce premium items for better returns."
#             ]

#         # Plot and save the forecast chart
#         plt.figure(figsize=(12, 6))
#         plt.plot(forecast['ds'].values, forecast['yhat'].values, label='Forecast', color='blue')
#         plt.fill_between(forecast['ds'].values,
#                          forecast['yhat_lower'].values,
#                          forecast['yhat_upper'].values,
#                          color='lightblue', alpha=0.5)
#         plt.xlabel('Date')
#         plt.ylabel('Price')
#         plt.title('Forecasted Prices')
#         plt.legend()

#         img_filename = f"{product}_forecast.png"
#         img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
#         plt.savefig(img_path)
#         plt.close()

#         return render_template("index.html",
#                        categories=list(category_product_map.keys()),
#                        product_map=category_product_map,
#                        selected_category=category,
#                        selected_product=product,
#                        avg_price=avg_price,
#                        peak_price=peak_price,
#                        forecast_img=img_filename,
#                        buyer_insights=buyer_insights,
#                        seller_insights=seller_insights,
#                        trend_message=trend_message)

#     else:
#         return render_template("index.html",
#                        categories=list(category_product_map.keys()),
#                        product_map=category_product_map,
#                        selected_category=None,
#                        selected_product=None)
                       
# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/help')
# def help_page():
#     return render_template('help.html')

# if __name__ == '__main__':
#     app.run(debug=True)


#Updated code with error handling and logging
from flask import Flask, render_template, request
import os
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
import logging

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'images')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure directory exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_product_data():
    """Load and preprocess product data with error handling"""
    try:
        # Try multiple encodings to handle different CSV formats
        encodings = ['utf-8', 'latin1', 'cp1252']
        for encoding in encodings:
            try:
                product_data = pd.read_csv('price/preprocessed_data.csv', encoding=encoding)
                # Convert date column to datetime if needed
                if 'Date' in product_data.columns:
                    product_data['Date'] = pd.to_datetime(product_data['Date'])
                return product_data
            except UnicodeDecodeError:
                continue
        raise ValueError("Failed to read CSV file with tried encodings")
    except Exception as e:
        logger.error(f"Error loading product data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

# Load product data at startup
product_data = load_product_data()
if not product_data.empty:
    category_product_map = (
        product_data.groupby('Category')['Product']
        .unique().apply(list).to_dict()
    )
else:
    category_product_map = {}
    logger.warning("Product data is empty - category_product_map will be empty")

@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    if request.method == 'POST':
        try:
            category = request.form.get('category', '').strip()
            product = request.form.get('product', '').strip()

            if not category or not product:
                return render_template("index.html",
                                    categories=list(category_product_map.keys()),
                                    product_map=category_product_map,
                                    error="Please select both category and product.")

            # Filter data for selected product
            df = product_data[(product_data['Category'] == category) & 
                            (product_data['Product'] == product)].copy()

            if df.empty:
                return render_template("index.html",
                                    categories=list(category_product_map.keys()),
                                    product_map=category_product_map,
                                    error="No data available for selected category and product.")

            # Prepare data for Prophet
            df = df[['Date', 'Price']].rename(columns={'Date': 'ds', 'Price': 'y'})
            df['ds'] = pd.to_datetime(df['ds'])
            df = df.dropna()  # Remove any NA values

            # Handle cases with insufficient data
            if len(df) < 2:
                return render_template("index.html",
                                    categories=list(category_product_map.keys()),
                                    product_map=category_product_map,
                                    error="Insufficient data points for forecasting.")

            # Create and fit model
            model = Prophet(daily_seasonality=False)
            model.fit(df)
            
            # Generate forecast (30 days into future)
            future = model.make_future_dataframe(periods=30)
            forecast_df = model.predict(future)

            # Calculate metrics
            avg_price = round(forecast_df['yhat'].mean(), 2)
            peak_price = round(forecast_df['yhat'].max(), 2)

            # Determine price trend direction
            trend_start = forecast_df['yhat'].iloc[-31]
            trend_end = forecast_df['yhat'].iloc[-1]
            is_trend_decreasing = trend_end < trend_start

            trend_message = "🔻 The product price trend is decreasing." if is_trend_decreasing else "🔺 The product price trend is increasing."

            # Generate insights
            if is_trend_decreasing:
                buyer_insights = [
                    "Better Deals Ahead: Prices are expected to drop, consider waiting to purchase later.",
                    "Smart Budgeting: Plan purchases for when prices become more affordable.",
                    "Avoid Overpaying: Reduce chance of buying at high prices before a dip."
                ]
                seller_insights = [
                    "Adjust Pricing: Consider lowering prices to stay competitive.",
                    "Inventory Planning: Clear stock faster before values decrease.",
                    "Promotional Timing: Run offers early to maintain margins."
                ]
            else:
                buyer_insights = [
                    "Buy Now: Purchasing early could save money as prices rise.",
                    "Prioritize Essentials: Get important products before costs increase.",
                    "Early Deals: Seek discounts before prices go higher."
                ]
                seller_insights = [
                    "Higher Profits: Benefit from upcoming higher price points.",
                    "Strategic Stock: Time sales to align with peak values.",
                    "Premium Launch: Good time to introduce higher-priced items."
                ]

            # Create and save plot
            plt.figure(figsize=(12, 6))
            plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast', color='blue')
            plt.fill_between(forecast_df['ds'],
                            forecast_df['yhat_lower'],
                            forecast_df['yhat_upper'],
                            color='lightblue', alpha=0.3)
            plt.scatter(df['ds'], df['y'], color='red', label='Historical Prices', s=30)
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.title(f'Price Forecast for {product}')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_filename = f"{product.replace(' ', '_')}_forecast_{timestamp}.png"
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close()

            return render_template("index.html",
                                categories=list(category_product_map.keys()),
                                product_map=category_product_map,
                                selected_category=category,
                                selected_product=product,
                                avg_price=avg_price,
                                peak_price=peak_price,
                                forecast_img=os.path.join('images', img_filename),
                                buyer_insights=buyer_insights,
                                seller_insights=seller_insights,
                                trend_message=trend_message)

        except Exception as e:
            logger.error(f"Error in forecast route: {str(e)}")
            return render_template("index.html",
                                categories=list(category_product_map.keys()),
                                product_map=category_product_map,
                                error="An error occurred while processing your request. Please try again.")

    # GET request handling
    return render_template("index.html",
                         categories=list(category_product_map.keys()),
                         product_map=category_product_map,
                         selected_category=None,
                         selected_product=None)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help_page():
    return render_template('help.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)