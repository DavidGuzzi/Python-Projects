import pandas as pd

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo

app_ui = ui.page_fluid(
    ui.input_file("file1", "Choose CSV File", accept=[".csv", ".xlsx"], multiple=False),
    ui.output_data_frame("summary"),
)

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.calc
    def parsed_file():
        file: list[FileInfo] | None = input.file1()
        if file is None:
            return pd.DataFrame()
        return pd.read_excel(  # pyright: ignore[reportUnknownMemberType]
            file[0]["datapath"]
        )

    @render.data_frame
    def summary():
        df = parsed_file()
        return df
    

app = App(app_ui, server)