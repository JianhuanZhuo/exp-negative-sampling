from class_resolver import Resolver

from model.GMF import GMF

model_resolver = Resolver(
    {
        GMF,
    },
    base=object,  # type: ignore
    default=GMF,
    suffix='',
)
