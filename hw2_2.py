import marimo

__generated_with = "0.17.8"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # HomeWork 1
    **Team 6:**
    Petrosyan Karolina,
    Shitova Marina,
    Kuznetsov Vladimir
    """)
    return


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    return pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **Data preparation**
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv('all_v2.csv')
    return (df,)


@app.cell
def _(df):
    df_1 = df.drop_duplicates()
    return (df_1,)


@app.cell
def _(df_1, pd):
    df_2 = df_1[df_1['price'] > 0]
    df_2 = df_2[df_2['rooms'] > 0]
    df_2 = df_2[(df_2['area'] >= 10) & (df_2['area'] <= 500)]
    df_2 = df_2[(df_2['kitchen_area'] >= 2) & (df_2['kitchen_area'] <= 0.5 *df_2['area'])]

    df_2['building_type'] = df_2['building_type'].astype(str)
    df_2.loc[df_2['building_type'] == '0', 'building_type'] = 'Other'
    df_2.loc[df_2['building_type'] == '1', 'building_type'] = 'Panel'
    df_2.loc[df_2['building_type'] == '2', 'building_type'] = 'Monolithic'
    df_2.loc[df_2['building_type'] == '3', 'building_type'] = 'Brick'
    df_2.loc[df_2['building_type'] == '4', 'building_type'] = 'Blocky'
    df_2.loc[df_2['building_type'] == '5', 'building_type'] = 'Wooden'

    df_2['object_type'] = df_2['object_type'].astype(str)
    df_2.loc[df_2['object_type'] == '1', 'object_type'] = 'seconadary'
    df_2.loc[df_2['object_type'] == '11', 'object_type'] = 'new_building'

    df_2['datetime_str'] = df_2['date'].astype(str) + ' ' + df_2['time'].astype(str)
    df_2['datetime'] = pd.to_datetime(df_2['datetime_str'])
    df_2['hour'] = df_2['datetime'].dt.hour
    df_2['month'] = df_2['datetime'].dt.month
    df_2['year'] = df_2['datetime'].dt.year

    df_2 = df_2.drop(columns=['date', 'time', 'datetime_str', 'datetime'])
    return (df_2,)


@app.cell
def _(df_2):
    df_3 = df_2[df_2['level'] <= df_2['levels']]

    df_3 = df_3.drop(columns=['geo_lat', 'geo_lon'])
    return (df_3,)


@app.cell
def _(df_3, pd):
    df_4= df_3[df_3['price'] < df_3['price'].quantile(0.99)] #droppin extremal numbers to preven anomalies
    df_4 = df_4[df_4['price'] > df_4['price'].quantile(0.01)]

    df_4 = pd.get_dummies(df_4)
    return (df_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## **ML**
    """)
    return


@app.cell
def _(df_4):
    from sklearn.model_selection import train_test_split
    X = df_4.drop(columns=['price'])
    y = df_4['price']

    train_X, holdout_X, train_y, holdout_y = train_test_split(X, y, test_size=0.4, random_state=1)
    return train_X, train_y


@app.cell
def _():
    import joblib

    model1 = joblib.load("model_3.pkl")

    return (model1,)


@app.cell
def _():
    region_map_second = {
        "2661": "Санкт-Петербург",
        "3446": "Ленинградская область",
        "3": "Москва",
        "81": "Московская область",
        "2843": "Краснодарский край",
        "2871": "Нижегородская область",
        "3230": "Ростовская область",
        "3106": "Самарская область",
        "2922": "Республика Татарстан",
        "2900": "Ставропольский край",
        "2722": "Республика Башкортостан",
        "6171": "Свердловская область",
        "4417": "Республика Коми",
        "5282": "Челябинская область",
        "5368": "Иркутская область",
        "5520": "Пермский край",
        "6817": "Алтайский край",
        "9579": "Республика Бурятия",
        "2604": "Ярославская область",
        "1010": "Удмуртская Республика",
        "7793": "Псковская область",
        "13919": "Республика Северная Осетия — Алания",
        "2860": "Кемеровская область",
        "3019": "Чувашская Республика",
        "4982": "Республика Марий Эл",
        "9648": "Кабардино-Балкарская Республика",
        "5241": "Республика Мордовия",
        "3870": "Красноярский край",
        "3991": "Тюменская область",
        "2359": "Республика Хакасия",
        "9654": "Новосибирская область",
        "2072": "Воронежская область",
        "8090": "Республика Карелия",
        "4007": "Республика Дагестан",
        "11171": "Республика Саха (Якутия)",
        "10160": "Забайкальский край",
        "7873": "Республика Крым",
        "6937": "Республика Крым",
        "2594": "Кировская область",
        "8509": "Республика Калмыкия",
        "11416": "Республика Адыгея",
        "11991": "Карачаево-Черкесская Республика",
        "5178": "Республика Тыва",
        "13913": "Республика Ингушетия",
        "6309": "Республика Алтай",
        "5952": "Белгородская область",
        "6543": "Архангельская область",
        "2880": "Тверская область",
        "5993": "Пензенская область",
        "2484": "Ханты-Мансийский автономный округ",
        "4240": "Липецкая область",
        "5789": "Владимирская область",
        "14880": "Ямало-Ненецкий автономный округ", 
        "1491": "Рязанская область",
        "2885": "Чеченская Республика",
        "5794": "Смоленская область",
        "2528": "Саратовская область",
        "4374": "Вологодская область",
        "4695": "Волгоградская область",
        "2328": "Калужская область",
        "5143": "Тульская область",
        "2806": "Тамбовская область",
        "14368": "Мурманская область",
        "5736": "Новгородская область",
        "7121": "Курская область",
        "4086": "Хабаровский край",
        "821": "Брянская область",
        "10582": "Астраханская область",
        "7896": "Калининградская область",
        "8640": "Омская область",
        "5703": "Курганская область",
        "10201": "Томская область",
        "4249": "Ульяновская область",
        "3153": "Оренбургская область",
        "4189": "Костромская область",
        "2814": "Орловская область",
        "13098": "Камчатский край",
        "8894": "Ивановская область",
        "7929": "Амурская область",
        "16705": "Магаданская область",
        "69": "Еврейская автономная область",
        "4963": "Приморский край",
        "1901": "Сахалинская область",
        "61888": "Ненецкий автономный округ",
    }

    region_map_second = dict(sorted(region_map_second.items(), key=lambda item: item[1]))

    region_code_by_name = {name: int(code) for code, name in region_map_second.items()}
    return (region_code_by_name,)


@app.cell
def _():
    building_type_dict = {
            "Other": "Other",
            "Panel": "Panel",
            "Monolithic": "Monolithic",
            "Brick": "Brick",
            "Block": "Block",
            "Wooden": "Wooden"}

    object_type_dict={
            "Secondary": "seconadary",
            "New building": "new_building",
        }
    return building_type_dict, object_type_dict


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(building_type_dict, mo, object_type_dict, region_code_by_name):
    area_slider = mo.ui.slider(10, 250, 1,label="Total area, m²")
    rooms_slider = mo.ui.slider(-1, 10, 1, label="Rooms (-1 = studio)")
    level_slider = mo.ui.slider(1, 30, 1,  label="Floor")
    levels_slider = mo.ui.slider(1, 50, 1, label="Total floors")
    kitchen_slider = mo.ui.slider(2, 168, 1,label="Kitchen area, m²")
    region_dropdown = mo.ui.dropdown(
        options=region_code_by_name,   # ключи = названия, значения = коды
        value="Москва",                
        label="Region",
    )

    building_type_dropdown = mo.ui.dropdown(
        building_type_dict,
        value="Panel",
        label="Building type",
    )

    object_type_dropdown = mo.ui.dropdown(
        object_type_dict,
        value="Secondary",
        label="Object type",
    )

    mo.vstack(
        [
            mo.md("## **1. Set apartment parameters**"),
            mo.md(
                "Hello! Use the controls below to describe the apartment. The model will use these parameters to estimate the market price."),
            mo.hstack(
                [
                    mo.vstack([region_dropdown, building_type_dropdown, object_type_dropdown, rooms_slider], gap=1),
                    mo.vstack([area_slider, kitchen_slider, level_slider, levels_slider], gap=1),
                ],
                gap=5,  
            ),
        ],
        gap=1,
    )
    return (
        area_slider,
        building_type_dropdown,
        kitchen_slider,
        level_slider,
        levels_slider,
        object_type_dropdown,
        region_dropdown,
        rooms_slider,
    )


@app.cell
def _(
    area_slider,
    building_type_dropdown,
    kitchen_slider,
    level_slider,
    levels_slider,
    model1,
    object_type_dropdown,
    pd,
    region_dropdown,
    rooms_slider,
    train_X,
):
    feature_cols = list(train_X.columns)

    def build_features_row():
        row = {col: 0 for col in feature_cols}

        if "region" in row:
            row["region"] = region_dropdown.value 
        if "area" in row:
            row["area"] = area_slider.value
        if "rooms" in row:
            row["rooms"] = rooms_slider.value
        if "level" in row:
            row["level"] = level_slider.value
        if "levels" in row:
            row["levels"] = levels_slider.value
        if "kitchen_area" in row:
            row["kitchen_area"] = kitchen_slider.value

        bt_cols = [
            "building_type_Blocky",
            "building_type_Brick",
            "building_type_Monolithic",
            "building_type_Other",
            "building_type_Panel",
            "building_type_Wooden",
        ]
        for col in bt_cols:
            if col in row:
                row[col] = 0

        bt_value = building_type_dropdown.value        
        bt_col = f"building_type_{bt_value}"           
        if bt_col in row:
            row[bt_col] = 1


        ot_cols = ["object_type_new_building", "object_type_seconadary"]
        for col in ot_cols:
            if col in row:
                row[col] = 0

        ot_value = object_type_dropdown.value          
        ot_col = f"object_type_{ot_value}"
        if ot_col in row:
            row[ot_col] = 1


        return pd.DataFrame([row], columns=feature_cols)



    X_current = build_features_row()
    predicted_price = float(model1.predict(X_current)[0])
    return (predicted_price,)


@app.cell
def _(
    building_type_dropdown,
    object_type_dropdown,
    plt,
    predicted_price,
    region_dropdown,
    train_X,
    train_y,
):
    # Prepare offers_df 
    offers_df = train_X.copy()
    offers_df["price"] = train_y.values   

    # recover building_type and object_type from dummies
    bt_cols = [c for c in offers_df.columns if c.startswith("building_type_")]
    offers_df["building_type"] = (
        offers_df[bt_cols]
        .idxmax(axis=1)
        .str.replace("building_type_", "", regex=False)
    )

    ot_cols = ["object_type_new_building", "object_type_seconadary"]
    offers_df["object_type"] = (
        offers_df[ot_cols]
        .idxmax(axis=1)
        .str.replace("object_type_", "", regex=False)
    )

    # Price distribution in region
    def plot_price_distribution_region():
        region = region_dropdown.value
        our_price = predicted_price

        prices = offers_df.loc[offers_df["region"] == region, "price"]

        fig, ax = plt.subplots(figsize=(6, 3))

        n, bins, patches = ax.hist(prices, bins=30, edgecolor="black")

        import numpy as np
        bin_index = np.digitize(our_price, bins) - 1

        if 0 <= bin_index < len(patches):
            patches[bin_index].set_facecolor("red")
            patches[bin_index].set_alpha(0.8)

        ax.set_title("Price distribution in selected region")
        ax.set_xlabel("Price, RUB")
        ax.set_ylabel("Number of listings")
        fig.tight_layout()
        return fig



    # Popularity of (building_type, object_type) in region
    def plot_type_popularity_region():
        region = region_dropdown.value

        sel_bt = building_type_dropdown.value      
        sel_ot = object_type_dropdown.value    

        subset = offers_df[offers_df["region"] == region].copy()

        fig, ax = plt.subplots(figsize=(6, 3))

        counts = (subset
            .groupby(["building_type", "object_type"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        counts["label"] = counts["building_type"] + " / " + counts["object_type"]

        mask_selected = (
            (counts["building_type"] == sel_bt) &
            (counts["object_type"] == sel_ot)
        )

        bars = ax.bar(range(len(counts)), counts["count"])

        for i, bar in enumerate(bars):
            if mask_selected.iloc[i]:
                bar.set_facecolor("red")
                bar.set_alpha(0.8)

        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts["label"], rotation=45, ha="right")
        ax.set_ylabel("Number of listings")
        ax.set_title("Popularity of types in selected region")

        fig.tight_layout()
        return fig


    fig_prices = plot_price_distribution_region()
    fig_popularity = plot_type_popularity_region()
    return fig_popularity, fig_prices


@app.cell
def _(
    area_slider,
    building_type_dropdown,
    fig_popularity,
    fig_prices,
    kitchen_slider,
    level_slider,
    levels_slider,
    mo,
    object_type_dropdown,
    predicted_price,
    region_dropdown,
    rooms_slider,
):
    mo.hstack(
        [mo.vstack(
            [
                mo.md("## **2. Predicted price from ML model**"),
                mo.md("### **Chosen properties:**\n"),
                mo.md(
                    f"Region:  {region_dropdown.value}   \n"
                    f"Building type: {building_type_dropdown.value}   \n"
                    f"Object type: {object_type_dropdown.value}   \n"
                    f"Total area: {area_slider.value}   \n"
                    f"Rooms:  {rooms_slider.value}  \n"
                    f"Floor:  {level_slider.value}   \n"
                    f"Total floors: {levels_slider.value}  \n"
                    f"Kitchen area: {kitchen_slider.value}   \n"
                ),
                mo.md("### **Predicted price:**"),
                mo.md(f"{predicted_price:,.0f} RUB"), 
            ], gap=1),
         mo.vstack(
            [
                mo.md("### .        \n"),
                mo.md("### **Statistics:**\n"),
                fig_prices,
                fig_popularity
            ], gap=1)
        ], gap=2
    )
    return


if __name__ == "__main__":
    app.run()
