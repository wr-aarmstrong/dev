
import sys
from databricks.feature_engineering import FeatureEngineeringClient
from features import AsyncFeatureTable, PricingFeature

SPARK_SESSION_APP_NAME = "async_features"


def update_offline_table_schema_and_possibly_backfill(feature_object: AsyncFeatureTable,
                                                      df) -> None:
    fe = FeatureEngineeringClient()

    if feature_object.spark.catalog.tableExists(feature_object.table_name):
        table = fe.get_table(name=feature_object.table_name)
        if new_columns := set(df.columns) - set(table.features):
            #todo: update this section, add backfill
            print(f"Table has new columns: {new_columns}")
            print("todo: update table")

        assert set(table.primary_keys) == set(
            feature_object.primary_keys), f"{table.primary_keys=} and {feature_object.primary_keys=} are not the same"

        print("DEBUG: Table already exists")
    else:
        # Do I need to have `collected_timestamp` as primary key too?
        fe.create_table(
            name=feature_object.table_name,
            # unique table name (in case you re-run the notebook multiple times)
            primary_keys=feature_object.primary_keys,
            timestamp_keys="ts",
            schema=df.schema,
            description=feature_object.description
        )
        print(f"Created feature table {feature_object.table_name}")


def append_to_feature_table(name, df):
    print(f"merge to feature table {name}")
    fe = FeatureEngineeringClient()
    fe.write_table(name=name, df=df, mode="merge")


def main(feature_object: AsyncFeatureTable):
    df = feature_object.compute_features()
    update_offline_table_schema_and_possibly_backfill(feature_object, df)
    append_to_feature_table(feature_object.table_name, df)


if __name__ == "__main__":
    argument = sys.argv[1]
    if argument == "pricing_hourly":
        selected_features = PricingFeature()
    else:
        # Feel free to add other features here
        raise NotImplementedError(f"'{argument}' currently not available")

    main(selected_features)
