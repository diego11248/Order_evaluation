from easyeditor import BaseEditor, MEMITHyperParams

def load_memit(hparams_path: str) -> BaseEditor:
    """Load a MEMIT editor from an EasyEdit hparams yaml path."""
    hp = MEMITHyperParams.from_hparams(hparams_path)
    return BaseEditor.from_hparams(hp)

def apply_edits(editor: BaseEditor, requests, **kwargs):
    """Apply a list of edit requests with the loaded editor."""
    # requests example: [{"prompt": "...", "target_new": "..."}]
    return editor.edit(requests, **kwargs)
