from enum import Enum
import sys
from abc import ABC
from typing import List, Dict, Callable
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
from databricks.feature_engineering import FeatureEngineeringClient

SPARK_SESSION_APP_NAME = "async_features"


class CalculationInterval(Enum):
    # Use cron presets
    HOURLY = "@hourly"
    DAILY = "@daily"
    # you can also add more elaborate cron expressions, like "0 * * 1 *"


class AsyncFeatureTable(ABC):
    """This class corresponds to an offline feature table."""
    interval: CalculationInterval
    table_name: str
    primary_keys: List[str]
    timestamp_key: str
    description: str

    def __init__(self):
        self.spark = SparkSession.builder.appName(SPARK_SESSION_APP_NAME).getOrCreate()

    def compute_features(self) -> DataFrame:
        raise NotImplementedError


class PricingFeature(AsyncFeatureTable):
    table_name = "stg_dev_zepz.features.andreas_pricing"
    description = "Pricing Features"
    timestamp_key = "ts"
    non_ts_primary_keys = [
        "name",
        "from_currency",
        "to_currency",
        "from_country",
        "to_country",
        "pay_in_method",
        "pay_out_method",
        "send_amount",
    ]
    primary_keys = non_ts_primary_keys + [timestamp_key]

    def _get_fxc_table(self) -> DataFrame:
        return self.spark.table("raw_prd_sw.fivetran_fxc.raw_fxc")

    def prepare_fxc_table(self, df, n_last_days: int = 2) -> DataFrame:
        df = df.withColumnRenamed("collected_timestamp", "ts")

        # Excluding WorldRemit since that"s not a competitor anymore and would only create confusion
        df = df.filter(
            (F.col("ts") > F.date_sub(F.current_date(), n_last_days))
            & (df.customer_price_type.isin(["Not specified", "Repeat customer pricing"]))
            & (~df.name.isin(["WorldRemit", "Sendwave", "Worldremit Ltd."]))
        )

        df = df.withColumn(
            "effective_competitor_rate",
            (df.send_amount * df.fx_rate - df.fee_to_currency) / df.send_amount
        )
        return df.select(self.primary_keys + ["effective_competitor_rate"])

    def effective_competitor_rate(self) -> DataFrame:
        df = self._get_fxc_table()
        df = self.prepare_fxc_table(df, 2)
        w = Window.partitionBy(self.non_ts_primary_keys).orderBy(F.desc(self.timestamp_key))
        df2 = df.withColumn("Rank", F.dense_rank().over(w))
        df2 = df2.filter(df2.Rank == 1).drop(df2.Rank)
        return df2.select(
            self.primary_keys + ["effective_competitor_rate"]
        )

    def _effective_competitor_rate_avg_n_last_days(self, n_last_days: int) -> DataFrame:
        df = self._get_fxc_table()
        df = self.prepare_fxc_table(df, n_last_days)
        return (
            df.groupBy(self.primary_keys)
            .agg(
                F.mean("effective_competitor_rate")
                .alias(f"effective_competitor_rate_avg_{n_last_days}")
            )
        )

    def effective_competitor_rate_avg_3(self) -> DataFrame:
        return self._effective_competitor_rate_avg_n_last_days(3)

    def effective_competitor_rate_avg_5(self) -> DataFrame:
        return self._effective_competitor_rate_avg_n_last_days(5)

    def effective_competitor_rate_avg_7(self) -> DataFrame:
        return self._effective_competitor_rate_avg_n_last_days(7)

    def compute_features(self):
        df_total = None
        # todo: build a register (possibly with python decorators)
        for feature_function in [
            self.effective_competitor_rate,
            self.effective_competitor_rate_avg_3,
            self.effective_competitor_rate_avg_5,
            self.effective_competitor_rate_avg_7,
        ]:
            try:
                df = feature_function()
                df_total = df if df_total is None else df_total.join(df, self.primary_keys,
                                                                     "outer")
            except Exception as e:
                print(f"There is the exception {e}")

        return df_total


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
