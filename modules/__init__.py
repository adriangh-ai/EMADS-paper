from transformers import AutoModel
from transformers.pipelines import PIPELINE_REGISTRY
from modules.composition import CompositionPipeline

PIPELINE_REGISTRY.register_pipeline(
    "composition",
    pipeline_class=CompositionPipeline,
    pt_model=AutoModel,
)