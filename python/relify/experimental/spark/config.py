from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...builders.v1 import BuilderCapabilities, BuilderInfo, BuildProfile

if TYPE_CHECKING:
    from ...builders.v1 import BuildContext, BuildOutput, BuildRequest


SPARK_BUILDER_INFO = BuilderInfo("spark", "Apache Spark", "relify")
SPARK_BUILDER_CAPABILITIES = BuilderCapabilities(
    frozenset({BuildProfile("ivf", "iceberg", "iceberg")})
)


@dataclass(frozen=True)
class Spark:
    """Configure the experimental Apache Spark index builder."""

    spark: Any

    def __post_init__(self) -> None:
        try:
            context = self.spark.sparkContext
        except Exception as error:
            raise NotImplementedError(
                "the first Spark builder supports Spark Classic only"
            ) from error
        if context is None:
            raise TypeError("spark must be an active pyspark.sql.SparkSession")

    @property
    def info(self) -> BuilderInfo:
        return SPARK_BUILDER_INFO

    @property
    def capabilities(self) -> BuilderCapabilities:
        return SPARK_BUILDER_CAPABILITIES

    def build(
        self,
        request: BuildRequest,
        context: BuildContext,
    ) -> BuildOutput:
        import uuid

        from .builder import build_initial

        spark_context = self.spark.sparkContext
        group = f"relify-build-{request.index}-{uuid.uuid4().hex}"
        spark_context.setJobGroup(
            group,
            f"Build Relify index {request.index}",
            True,
        )
        try:
            return build_initial(self.spark, request, context)
        finally:
            for property_name in (
                "spark.jobGroup.id",
                "spark.job.description",
                "spark.job.interruptOnCancel",
            ):
                spark_context.setLocalProperty(property_name, None)
