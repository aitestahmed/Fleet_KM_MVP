def map_columns(df):

    column_map = {}

    for col in df.columns:

        c = str(col).lower()

        # kilometers
        if "كيلومتر" in c or "km" in c or "distance" in c or "trip" in c:
            column_map[col] = "kilometers"

        # fuel / expense
        elif "مصروف" in c or "cost" in c or "expense" in c or "fuel" in c:
            column_map[col] = "expense_amount"

        # vehicle
        elif "سيارة" in c or "vehicle" in c or "truck" in c:
            column_map[col] = "vehicle"

        # quantity
        elif "كمية" in c or "qty" in c or "quantity" in c:
            column_map[col] = "quantity"

        # product
        elif "منتج" in c or "product" in c:
            column_map[col] = "product"

    df = df.rename(columns=column_map)

    return df
