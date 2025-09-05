from easyeditor import BaseEditor, AlphaEditHyperParams

def load_alphaedit(hparams_path: str) -> BaseEditor:
    hp = AlphaEditHyperParams.from_hparams(hparams_path)
    return BaseEditor.from_hparams(hp)

def apply_edits(editor: BaseEditor, requests, **kwargs):
    return editor.edit(requests, **kwargs)
