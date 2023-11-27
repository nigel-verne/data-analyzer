"""
DAQ Data Processing and Analysis Program

Author: Nigel Veach
Date: 2023/09/07
First Revision of the Program
Currently compatible with AST1, MTS, & TSN Systems

This program is designed for processing and analyzing data acquired from various DAQ
systems.  It loads configuration settings from a JSON file, imports and cleans DAQ 
data files, calculates additional values, and generates interactive plots using Plotly.

Revision History:
2023/09/13 Draft code reviewed by Vincent Heloin.
2023/09/15 First revision of program completed.  Compatible with AST1, MTS, & TSN
2023/09/18 Fixed bug where sensor correction factors were applied multiple times
2023/09/19 Improved hoverlabel functionality, and exported file as executable
"""


import pandas as pd
import json
import plotly.graph_objs as go
import plotly.offline
import numpy as np
from pathlib import Path
from CoolProp.CoolProp import PropsSI


def main():
    """
    Main function for processing and analyzing DAQ data.

    Loads configuration, imports & cleans DAQ data files, calculates additional values,
    and generates plots.
    """
    # Load configuration from a JSON file
    config = load_config(Path("config.json"))
    if config is None:
        return

    # Extract program configuration variables
    settings = config.get("settings", {})
    system_name = settings.get("system_name", "")
    data_directory = Path(settings.get("data_directory", ""))
    derived_values_enabled = settings.get("derived_values_enabled", False)
    plotting_enabled = settings.get("plotting_enabled", False)
    merge_data_enabled = settings.get("merge_data_enabled", False)

    # Extract system-specific configuration variables
    system_specifications = config.get(system_name, {}).get("system_specifications", {})
    sensors = config.get(system_name, {}).get("sensors", {})
    values_to_derive = config.get(system_name, {}).get("values_to_derive", {})

    # Create an output directory by prepending "output_" to the data directory
    output_parent_directory = Path("output_" + str(data_directory))

    # Check if the data directory exists and is a directory
    if not data_directory.exists() or not data_directory.is_dir():
        print(f"The directory '{data_directory}' does not exist or is not a directory.")
        return

    # create an empty dataframe and legend to store the merged data
    merged_df = pd.DataFrame()
    merged_legend = {}

    # Iterate over all files in the data directory
    print("Loading DAQ Files...")
    for file in data_directory.rglob("*.[csv xlsx]*"):
        print(file)
        # Flatten sensor info so that it can be used to create a legend for the datafile
        legend = extract_leaf_key_values(sensors)

        # Read CSV or Excel file into a DataFrame
        df = pd.read_csv(file) if file.suffix == ".csv" else pd.read_excel(file)

        # Clean file
        df, legend = clean_dataframe(df, sensors, legend, file)

        # Define the new filename, replacing colons with periods to avoid naming issues
        merged_filename = str(df["Timestamp"].iloc[0]).replace(":", ".")

        # Define and create the new directory for the output file
        new_directory = (
            output_parent_directory
            / file.relative_to(data_directory).parent
            / merged_filename
        )
        new_directory.mkdir(parents=True, exist_ok=True)

        # Calculate derived values
        if derived_values_enabled:
            df, legend = calculate_derived_values(
                df,
                values_to_derive,
                system_specifications,
                legend,
                new_directory,
            )
            # Export processed data to file
            export_path = new_directory / (merged_filename + "_PROCESSED")
            export_file(df, export_path)
        else:
            export_path = new_directory / (merged_filename + "_CLEANED")
            export_file(df, export_path)

        # Generate plot
        if plotting_enabled:
            generate_plot(df, legend, system_name, new_directory, merged_filename)

        merged_df = pd.concat([merged_df, df], ignore_index=True, sort=False)
        merged_legend.update(
            {key: value for key, value in legend.items() if key not in merged_legend}
        )
        print("")

    if merge_data_enabled:
        merged_filename = "merged_data"

        # Define and create the new directory for the output file
        new_directory = output_parent_directory / merged_filename
        new_directory.mkdir(parents=True, exist_ok=True)

        # Clean file
        merged_df, merged_legend = clean_dataframe(
            merged_df, sensors, merged_legend, file=merged_filename
        )

        # Export merged data to file
        export_path = new_directory / merged_filename
        export_file(merged_df, export_path)

        # Generate merged data plot
        generate_plot(
            merged_df, merged_legend, system_name, new_directory, merged_filename
        )


def load_config(filename):
    """
    Load program configuration settings from a JSON file.

    Args:
        filename (Path): Path to the JSON configuration file.

    Returns:
        dict: A dictionary containing the program configuration settings.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If there is an issue decoding the JSON.
    """
    try:
        with open(filename, "r") as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError:
        print("Error: Config file not found.")
        raise  # Re-raise the exception
    except json.JSONDecodeError:
        print("Error: Unable to decode JSON in config file, check file for errors")
        raise  # Re-raise the exception


def extract_leaf_key_values(input_dict, parent_key=""):
    """
    Extract leaf key-value pairs from a nested dictionary, "flattening" it

    Args:
        input_dict (dict): The input nested dictionary.
        parent_key (str): The parent key for the current recursive call.

    Returns:
        dict: Dictionary containing only leaf key-value pairs.
    """
    leaf_key_values = {}

    for key, value in input_dict.items():
        new_key = f"{parent_key}.{key}" if parent_key else key

        if isinstance(value, dict):
            leaf_key_values.update(extract_leaf_key_values(value, new_key))
        else:
            leaf_key_values[key] = value

    return leaf_key_values


def clean_dataframe(df, sensors, legend, file):
    """
    Clean a given pandas DataFrame by performing the following operations:
    -removing colons from the filename
    -standardizing timestamp format
    -removing rows with duplicate or nonexistent timestamps
    -inserting an empty row at timestamp gaps longer than 1 minute
    -removing erroneous Temperature sensor values
    -updating the legend to remove any sensors whose data was not recorded

    Args:
        df (pd.DataFrame): The DataFrame containing test data.
        sensors (dict): Dictionary containing DAQ sensor information.
        legend (dict): Dictionary of possible data IDs and their labels.
        file (str): Name of the file being cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame containing test data.
    """
    # Display a message indicating which file's data is being cleaned
    print(f"Cleaning data for {file}")

    # Create a copy of the DataFrame to avoid SettingWithCopy Warning
    df_cleaned = df.copy()

    # remove spaces from the dataframe headers
    df_cleaned = df_cleaned.rename(columns=lambda x: x.replace(" ", ""))

    # Identify the Timestamp format and process accordingly
    if "Timestamp" in df_cleaned.columns:  # Handle timestamps for a preprocessed file
        df_cleaned["Timestamp"] = pd.to_datetime(df_cleaned["Timestamp"])
    elif "Date" in df_cleaned.columns:  # Handle timestamps for AST1 DAQ format
        df_cleaned.insert(
            0,
            "Timestamp",
            pd.to_datetime(df_cleaned["Date"] + " " + df_cleaned["Time"]),
        )
        df_cleaned = df_cleaned.drop(columns=["Date", "Time"])
    else:  # Handle timestamps for MTS DAQ format1
        df_cleaned.insert(
            0,
            "Timestamp",
            pd.to_datetime(df_cleaned["Time"], format="%m/%d/%Y %H:%M:%S"),
        )
        df_cleaned = df_cleaned.drop(columns=["Time"])

    # sort dataframe by ascending Timestamp
    df_cleaned = df_cleaned.sort_values(by="Timestamp", ascending=True)

    # clean dataframe by removing rows without timestamps
    df_cleaned = df_cleaned.dropna(subset=["Timestamp"])

    # clean dataframe by removing rows with duplicate timestamps
    df_cleaned = df_cleaned.drop_duplicates(subset="Timestamp")

    # reset the dataframe index
    df_cleaned = df_cleaned.reset_index(drop=True)

    # insert empty rows any time there is a timestamp gap greater than 1 minute
    # this ensures the plot has gaps where no DAQ data was collected
    time_diff = df_cleaned["Timestamp"].diff()
    gap_indices = np.where(time_diff > pd.Timedelta(minutes=1))[0]
    empty_rows = pd.DataFrame(np.NaN, columns=df_cleaned.columns, index=gap_indices)

    # set ["Timestamp"] to an empty string instead of np.Nan, so that sorting works
    empty_rows["Timestamp"] = ""

    # add the empty rows to the dataframe, resort the dataframe, and delete the index
    df_cleaned = pd.concat([empty_rows, df_cleaned])
    df_cleaned["index"] = df_cleaned.index
    df_cleaned = df_cleaned.sort_values(["index", "Timestamp"], ascending=[True, False])
    df_cleaned = df_cleaned.drop(columns=["index"])

    # DAQ records temperatures as high when dead and 31.13 when unplugged
    # Remove erroneous temperature sensor values (high and 31.13)
    for col_name in sensors["Temperature (K)"].keys():
        if col_name in df_cleaned.columns:
            df_cleaned.loc[df_cleaned[col_name] > 500, col_name] = np.NaN
            df_cleaned.loc[df_cleaned[col_name] == 31.13, col_name] = np.NaN
            df_cleaned.loc[df_cleaned[col_name] < 0, col_name] = np.NaN

    # filter the legend to remove any sensor values that were not actually recorded
    filtered_legend = {
        key: value for key, value in legend.items() if key in df_cleaned.columns
    }

    # Reset the DataFrame indices
    df_cleaned = df_cleaned.reset_index(drop=True)

    return df_cleaned, filtered_legend


def export_file(df, export_path):
    """
    Export a DataFrame to a file in either Excel (.xlsx) or CSV (.csv) format based on
    the row count.  If the DataFrame has fewer than 1,048,576 rows, it will be exported
    as an Excel file (.xlsx) to the specified export path. Otherwise, it will be
    exported as a CSV file (.csv) to the export path.  This is to avoid exceeding the
    row limit of Microsoft Excel.

    Args:
        df (pd.DataFrame): The DataFrame to be exported.
        export_path (pathlib.Path): The path where the exported file will be saved
    """
    print("Exporting processed data file to " + str(export_path))

    if df.shape[0] < 1048576:
        df.to_excel(Path(str(export_path) + ".xlsx"), index=False)
    else:
        df.to_csv(Path(str(export_path) + ".csv"), index=False)


def calculate_derived_values(
    df, values_to_derive, system_specifications, legend, directory
):
    """
    Calculate derived values from a DataFrame and update sensor configuration.

    Args:
        df (pd.DataFrame): The DataFrame containing test data.
        values_to_derive (dict): Dictionary specifying calculations to apply.
        system_specifications (dict): Dictionary containing system specifications.
        legend (dict): Dictionary of data IDs and their labels.
        directory (str): Identifier for the file's directory.

    Returns:
        df_copy (pd.DataFrame): The updated DataFrame with derived values.
        legend (dict): Updated dictionary of data IDs and their labels.
    """
    # Display a message indicating which calculations are being performed
    print(f"Calculating derived values for {directory}")

    # Create a copy of the DataFrame to avoid SettingWithCopy Warning
    df_copy = df.copy()

    """
    # Apply sensor correction factors to account for bad sensor calibrations
    if "Sensor Correction Factors" in values_to_derive:
        for sensor_ID, correction_factor in values_to_derive[
            "Sensor Correction Factors"
        ].items():
            df_copy[sensor_ID] = df_copy[sensor_ID] * correction_factor
    """

    # Calculate and add the average tank temperature as a new column ("AVG-T")
    if "Storage Average Temp (K)" in values_to_derive:
        avg_temp_columns = [
            col
            for col in values_to_derive["Storage Average Temp (K)"]
            if col in df_copy.columns
        ]
        df_copy["AVG-T"] = df_copy[avg_temp_columns].mean(axis=1)
        legend.update({"AVG-T": "Storage Average Temp (K)"})

    # Calculate and add hydrogen-related properties to the DataFrame
    if "Hydrogen Density (g/L)" in values_to_derive:
        temperature = values_to_derive["Hydrogen Density (g/L)"]["Temperature Data"]
        pressure = values_to_derive["Hydrogen Density (g/L)"]["Pressure Data"]

        # Calculate Hydrogen Density (g/L) and update plot legend accordingly
        df_copy["H2_Density"] = df_copy.apply(
            lambda row: get_hydrogen_property("D", row[temperature], row[pressure]),
            axis=1,
        )
        legend.update({"H2_Density": "Hydrogen Density (g/L)"})

        # Calculate Hydrogen Enthalpy (kJ/kg)
        df_copy["H2_Enthalpy"] = (
            df_copy.apply(
                lambda row: get_hydrogen_property("H", row[temperature], row[pressure]),
                axis=1,
            )
            / 1000
        )
        # Uncomment this line to plot hydrogen enthalpy
        # legend.update({"H2_Enthalpy": "Hydrogen Enthalpy (kJ/kg)"})

        # Calculate Hydrogen Entropy (kJ/(kg*K))
        df_copy["H2_Entropy"] = (
            df_copy.apply(
                lambda row: get_hydrogen_property("S", row[temperature], row[pressure]),
                axis=1,
            )
            / 1000
        )
        # Uncomment this line to plot hydrogen entropy
        # legend.update({"H2_Entropy": "Hydrogen Entropy (kJ/(kg*K))"})

        # Calculate Hydrogen Internal Energy (MJ/kg)
        df_copy["H2_Energy"] = (
            df_copy.apply(
                lambda row: get_hydrogen_property("U", row[temperature], row[pressure]),
                axis=1,
            )
            / 1000000
        )
        # Uncomment this line to plot hydrogen internal energy
        legend.update({"H2_Energy": "Hydrogen Internal Energy (MJ/kg)"})

        # Calculate Hydrogen Stored (kg) and update legend accordingly
        df_copy["H2_Stored"] = (
            df_copy["H2_Density"] * system_specifications["Tank Volume (L)"] / 1000
        )
        legend.update({"H2_Stored": "Hydrogen Stored (kg)"})

    # Calculate and add hydrogen-related properties to the DataFrame
    if "Inlet Hydrogen Density (g/L)" in values_to_derive:
        temperature = values_to_derive["Inlet Hydrogen Density (g/L)"][
            "Temperature Data"
        ]
        pressure = values_to_derive["Inlet Hydrogen Density (g/L)"]["Pressure Data"]

        # Calculate Hydrogen Density (g/L) and update plot legend accordingly
        df_copy["Inlet_H2_Density"] = df_copy.apply(
            lambda row: get_hydrogen_property("D", row[temperature], row[pressure]),
            axis=1,
        )
        legend.update({"Inlet_H2_Density": "Inlet Hydrogen Density (g/L)"})

        # Calculate Hydrogen Enthalpy (kJ/kg)
        df_copy["Inlet_H2_Enthalpy"] = (
            df_copy.apply(
                lambda row: get_hydrogen_property("H", row[temperature], row[pressure]),
                axis=1,
            )
            / 1000
        )
        # Uncomment this line to plot hydrogen enthalpy
        # legend.update({"Inlet_H2_Enthalpy": "Inlet Hydrogen Enthalpy (kJ/kg)"})

        # Calculate Hydrogen Entropy (kJ/(kg*K))
        df_copy["Inlet_H2_Entropy"] = (
            df_copy.apply(
                lambda row: get_hydrogen_property("S", row[temperature], row[pressure]),
                axis=1,
            )
            / 1000
        )
        # Uncomment this line to plot hydrogen entropy
        # legend.update({"Inlet_H2_Entropy": "Inlet Hydrogen Entropy (kJ/(kg*K))"})

        # Calculate Hydrogen Internal Energy (MJ/kg)
        df_copy["Inlet_H2_Energy"] = (
            df_copy.apply(
                lambda row: get_hydrogen_property("U", row[temperature], row[pressure]),
                axis=1,
            )
            / 1000000
        )
        # Uncomment this line to plot hydrogen internal energy
        legend.update({"Inlet_H2_Energy": "Inlet Hydrogen Internal Energy (MJ/kg)"})

    # Calculate and add hydrogen-related properties to the DataFrame
    if "Outlet Hydrogen Density (g/L)" in values_to_derive:
        temperature = values_to_derive["Outlet Hydrogen Density (g/L)"][
            "Temperature Data"
        ]
        pressure = values_to_derive["Outlet Hydrogen Density (g/L)"]["Pressure Data"]

        # Calculate Hydrogen Density (g/L) and update plot legend accordingly
        df_copy["Outlet_H2_Density"] = df_copy.apply(
            lambda row: get_hydrogen_property("D", row[temperature], row[pressure]),
            axis=1,
        )
        legend.update({"Outlet_H2_Density": "Outlet Hydrogen Density (g/L)"})

        # Calculate Hydrogen Enthalpy (kJ/kg)
        df_copy["Outlet_H2_Enthalpy"] = (
            df_copy.apply(
                lambda row: get_hydrogen_property("H", row[temperature], row[pressure]),
                axis=1,
            )
            / 1000
        )
        # Uncomment this line to plot hydrogen enthalpy
        # legend.update({"Outlet_H2_Enthalpy": "Outlet Hydrogen Enthalpy (kJ/kg)"})

        # Calculate Hydrogen Entropy (kJ/(kg*K))
        df_copy["Outlet_H2_Entropy"] = (
            df_copy.apply(
                lambda row: get_hydrogen_property("S", row[temperature], row[pressure]),
                axis=1,
            )
            / 1000
        )
        # Uncomment this line to plot hydrogen entropy
        # legend.update({"Outlet_H2_Entropy": "Outlet Hydrogen Entropy (kJ/(kg*K))"})

        # Calculate Hydrogen Internal Energy (MJ/kg)
        df_copy["Outlet_H2_Energy"] = (
            df_copy.apply(
                lambda row: get_hydrogen_property("U", row[temperature], row[pressure]),
                axis=1,
            )
            / 1000000
        )
        # Uncomment this line to plot hydrogen internal energy
        legend.update({"Outlet_H2_Energy": "Outlet Hydrogen Internal Energy (MJ/kg)"})

    # Round the DataFrame to handle float roundoff issues caused by CoolProp calcs
    df_copy = df_copy.round(decimals=4)

    return df_copy, legend


def get_hydrogen_property(property, temperature, pressure):
    """
    Calculates a provided property of hydrogen gas at a given temperature and pressure
    using CoolProp.  If either the temperature or pressure is NaN (Not-a-Number) or
    zero, the function returns NaN as the result.

    Args:
        property (char): The desired property of hydrogen:
            "D": density, kg/m^3
            "H": enthalpy, J/kg
            "S": entropy, J/(kg*K)
            "U": internal energy, J/kg
        temperature (float): The temperature in Kelvin.
        pressure (float): The pressure in Bar.

    Returns:
        float: The specified property of hydrogen gas in SI units
    """
    # Check if temperature or pressure is NaN or zero
    if np.isnan(temperature) or temperature == 0 or np.isnan(pressure) or pressure <= 0:
        return np.nan
    else:
        # Convert pressure from gauge pressure in bar to absolute pressure in Pascal
        # (CoolProp requires pressure in Pa)
        pressure_pa_gauge = (pressure + 1.01325) * 100000
        # Calculate and return the density in g/l
        return PropsSI(
            property, "T", temperature, "P", pressure_pa_gauge, "parahydrogen"
        )


def generate_plot(df, legend, system, directory, filename):
    """
    Generates and saves a data plot as an intactive html figure using plotly.

    Args:
        df (pd.DataFrame): DataFrame containing test data.
        legend (dict): Dictionary of data IDs and their labels.
        system (str): Name of the test system.
        directory (str): Directory path for saving plots.
        filename (str): Name of the output plot file.

    Returns:
        None
    """
    # Display a message indicating the file for which plots are being created
    print(f"Creating plots for {directory}")

    # Create list to store traces
    traces = []

    # Iterate through data to create plot traces
    for data_ID, data_description in legend.items():
        trace = go.Scatter(
            x=df["Timestamp"],
            y=df[data_ID],
            name=f"{data_ID}: {data_description}",
            connectgaps=False,
        )
        traces.append(trace)

    # Create the layout for the plot with title and subtitle
    title = f"{system} Testing: All Sensor Data"
    subtitle = filename
    layout = {
        "title": {
            "text": f"{title}<br><sub>{subtitle}</sub>",
            "x": 0.5,
            "xanchor": "center",
        },
        "xaxis": {"title": "Time", "showgrid": True, "gridcolor": "LightGray"},
        "yaxis": {"showgrid": True, "gridcolor": "LightGray"},
        "hovermode": "x unified",
        "hoverlabel": {
            "namelength": -1,  # Allow unlimited length for label text
        },
    }

    # Create plotly figure
    fig = go.Figure({"data": traces, "layout": layout})

    # Configure "Download as PNG" button settings and make plot editable
    config = {
        "toImageButtonOptions": {
            "format": "png",  # one of png, svg, jpeg, webp
            "filename": filename,
            "scale": 2,  # scale image by factor of 2 for higher resolution downloads
        },
        "editable": True,  # allow editing of text in figure
        "edits": {
            "legendPosition": False,
            "shapePosition": False,
        },  # disable moving legend position or changing graph data
    }

    # Export figure as interactive html graph
    plotly.offline.plot(
        fig,
        filename=str(directory / (filename + " All Data.html")),
        auto_open=False,
        config=config,
    )


if __name__ == "__main__":
    main()
