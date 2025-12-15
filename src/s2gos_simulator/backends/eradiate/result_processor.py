"""Result processing and visualization for Eradiate backend."""

from typing import Any, Dict

import numpy as np
import xarray as xr
from PIL import Image
from s2gos_utils.io.paths import mkdir, open_file
from upath import UPath

from .constants import RGB_WAVELENGTHS_NM

try:
    from eradiate.xarray.interp import dataarray_to_rgb

    ERADIATE_AVAILABLE = True
except ImportError:
    ERADIATE_AVAILABLE = False


class ResultProcessor:
    """Processor for simulation results, outputs, and visualizations."""

    def __init__(self, simulation_config):
        """Initialize result processor.

        Args:
            simulation_config: SimulationConfig object
        """
        self.simulation_config = simulation_config

    def process_results(self, results, output_dir: UPath) -> xr.Dataset:
        """Process and save simulation results.

        Args:
            experiment: Eradiate experiment with results
            output_dir: Directory for saving results

        Returns:
            Results dataset (dictionary or single dataset)

        Raises:
            ValueError: If no results found in experiment
        """

        if not results:
            raise ValueError("No results found in experiment")

        output_dir = UPath(output_dir)
        mkdir(output_dir)

        metadata = self.create_output_metadata(output_dir)

        if isinstance(results, dict):
            print(f"Processing {len(results)} measure results...")
            for sensor_id, dataset in results.items():
                sensor_output = (
                    output_dir / f"{self.simulation_config.name}_{sensor_id}.zarr"
                )

                dataset.attrs.update(metadata)
                dataset.attrs["sensor_id"] = sensor_id

                dataset.to_zarr(sensor_output, mode="w")
                print(f"Measure '{sensor_id}' saved to {sensor_output}")

        else:
            results_ds = results
            single_output = output_dir / f"{self.simulation_config.name}_results.zarr"
            results_ds.attrs.update(metadata)
            results_ds.to_zarr(single_output, mode="w")
            print(f"Results saved to {single_output}")

        return results

    def create_output_metadata(self, output_dir: UPath) -> Dict[str, Any]:
        """Create standardized metadata for output files.

        Args:
            output_dir: Output directory path

        Returns:
            Dictionary with metadata fields
        """
        return {
            "simulation_name": self.simulation_config.name,
            "description": self.simulation_config.description,
            "created_at": self.simulation_config.created_at.isoformat(),
            "backend": "eradiate",
            "output_dir": str(output_dir),
            "num_sensors": len(self.simulation_config.sensors),
            "num_measurements": len(self.simulation_config.measurements),
            "sensor_types": [
                s.platform_type.value for s in self.simulation_config.sensors
            ],
            "measurement_types": [m.type for m in self.simulation_config.measurements],
            "illumination_type": self.simulation_config.illumination.type,
        }

    def create_rgb_visualization(self, experiment, output_dir: UPath, id_to_plot: str):
        """Create RGB visualization from camera results.

        Args:
            experiment: Eradiate experiment with results
            output_dir: Output directory for images
            id_to_plot: Sensor ID to visualize
        """
        try:
            if id_to_plot not in experiment.results:
                print(f"Warning: Sensor '{id_to_plot}' not found in results")
                return

            sensor_data = experiment.results[id_to_plot]

            if "radiance" in sensor_data:
                radiance_data = sensor_data["radiance"]
                if "x_index" in radiance_data.dims and "y_index" in radiance_data.dims:
                    wavelengths = radiance_data.coords["w"].values

                    if len(wavelengths) >= 3:
                        target_wavelengths = RGB_WAVELENGTHS_NM
                        actual_wavelengths = [
                            radiance_data.sel(w=w_val, method="nearest").w.item()
                            for w_val in target_wavelengths
                        ]
                        corrected_channels = [
                            ("w", w_val) for w_val in actual_wavelengths
                        ]

                        img = (
                            dataarray_to_rgb(
                                radiance_data,
                                channels=corrected_channels,
                                normalize=False,
                            )
                            * 1.8
                        )
                        img = np.clip(img, 0, 1)
                        rgb_output = output_dir / f"{id_to_plot}_rgb.png"
                        plt_img = (img * 255).astype(np.uint8)
                        print(f"RGB image saved to: {rgb_output}")
                    else:
                        img_data = radiance_data.squeeze().values
                        img_normalized = (img_data - img_data.min()) / (
                            img_data.max() - img_data.min()
                        )
                        img_normalized = np.clip(img_normalized, 0, 1)
                        plt_img = (img_normalized * 255).astype(np.uint8)
                        rgb_output = output_dir / f"{id_to_plot}_grayscale.png"
                        print(f"Grayscale image saved to: {rgb_output}")

                    rgb_image = Image.fromarray(plt_img)
                    with open_file(rgb_output, "wb") as f:
                        rgb_image.save(f, format="PNG")

                else:
                    spectral_output = output_dir / f"{id_to_plot}_spectrum.png"
                    self.plot_spectral_data(radiance_data, spectral_output)
                    print(f"Spectral data plot saved to: {spectral_output}")

        except Exception as e:
            print(f"Warning: Could not create visualization for {id_to_plot}: {e}")

    def plot_spectral_data(self, radiance_data, output_path: UPath):
        """Plot spectral data for point sensors.

        Args:
            radiance_data: xarray DataArray with spectral radiance
            output_path: Path for output PNG file
        """
        try:
            import matplotlib.pyplot as plt

            wavelengths = radiance_data.coords["w"].values
            radiance_values = radiance_data.values

            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, radiance_values, "b-", linewidth=2)
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Radiance")
            plt.title("Spectral Radiance")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            with open_file(output_path, "wb") as f:
                plt.savefig(f, format="png", dpi=150, bbox_inches="tight")
            plt.close()

        except ImportError:
            print("Warning: matplotlib not available for spectral plotting")
        except Exception as e:
            print(f"Warning: Could not create spectral plot: {e}")

    def create_hdrf_visualizations(
        self, hdrf_results: Dict[str, xr.Dataset], output_dir: UPath
    ) -> None:
        """Create visualizations for HDRF results.

        Args:
            hdrf_results: Dictionary of HDRF datasets
            output_dir: Output directory for visualizations
        """
        import matplotlib.pyplot as plt

        vis_dir = output_dir / "hdrf_visualizations"
        mkdir(vis_dir)

        for measure_id, dataset in hdrf_results.items():
            try:
                hdrf_data = dataset["hdrf"]

                if "x_index" in hdrf_data.dims and "y_index" in hdrf_data.dims:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                    if "w" in hdrf_data.dims and len(hdrf_data.w) > 1:
                        img_data = hdrf_data.isel(w=0).values
                    else:
                        img_data = hdrf_data.squeeze().values

                    im = axes[0].imshow(img_data, cmap="RdYlGn", vmin=0, vmax=1)
                    axes[0].set_title(f"HDRF - {measure_id}")
                    axes[0].set_xlabel("X index")
                    axes[0].set_ylabel("Y index")
                    plt.colorbar(im, ax=axes[0], label="HDRF")

                    axes[1].hist(img_data.flatten(), bins=50, edgecolor="black")
                    axes[1].set_xlabel("HDRF")
                    axes[1].set_ylabel("Frequency")
                    axes[1].set_title("HDRF Distribution")
                    axes[1].axvline(
                        x=img_data.mean(), color="r", linestyle="--", label="Mean"
                    )
                    axes[1].legend()

                    plt.tight_layout()
                    output_file = vis_dir / f"{measure_id}_hdrf_visualization.png"
                    plt.savefig(output_file, dpi=150, bbox_inches="tight")
                    plt.close()

                    print(f"  Saved HDRF visualization: {output_file.name}")

                elif "w" in hdrf_data.dims:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    wavelengths = hdrf_data.w.values
                    hdrf_values = hdrf_data.values

                    ax.plot(wavelengths, hdrf_values, "b-", linewidth=2)
                    ax.set_xlabel("Wavelength (nm)")
                    ax.set_ylabel("HDRF")
                    ax.set_title(f"Spectral HDRF - {measure_id}")
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim([0, 1])

                    plt.tight_layout()
                    output_file = vis_dir / f"{measure_id}_hdrf_spectrum.png"
                    plt.savefig(output_file, dpi=150, bbox_inches="tight")
                    plt.close()

                    print(f"  Saved HDRF spectrum: {output_file.name}")

            except Exception as e:
                print(
                    f"  Warning: Could not create visualization for {measure_id}: {e}"
                )

    def create_dummy_radiative_quantity_result(
        self, rad_quantity, output_dir: UPath, metadata: dict
    ) -> UPath:
        """Create dummy Zarr file for radiative quantity placeholder.

        Args:
            rad_quantity: Radiative quantity configuration
            output_dir: Output directory
            metadata: Metadata dictionary

        Returns:
            Path to created dummy Zarr file
        """
        quantity_id = f"{rad_quantity.quantity.value}_measure"
        dummy_output = output_dir / f"{self.simulation_config.name}_{quantity_id}.zarr"

        dummy_data = np.ones((10, 10)) * 0.5

        wavelengths = [550.0]
        srf = rad_quantity.srf
        if srf.type == "delta" and srf.wavelengths:
            wavelengths = srf.wavelengths

        if len(wavelengths) > 1:
            dummy_values = np.stack([dummy_data for _ in wavelengths], axis=0)
            coords = {
                "w": wavelengths,
                "x": range(dummy_data.shape[0]),
                "y": range(dummy_data.shape[1]),
            }
            dims = ["w", "x", "y"]
        else:
            dummy_values = dummy_data
            coords = {"x": range(dummy_data.shape[0]), "y": range(dummy_data.shape[1])}
            dims = ["x", "y"]

        dummy_ds = xr.Dataset(
            {rad_quantity.quantity.value: (dims, dummy_values)}, coords=coords
        )

        dummy_ds.attrs.update(metadata)
        dummy_ds.attrs["quantity_id"] = quantity_id
        dummy_ds.attrs["note"] = "DUMMY DATA - placeholder for future implementation"

        dummy_ds.to_zarr(dummy_output, mode="w")
        print(f"Dummy data for '{quantity_id}' saved to {dummy_output}")

        return dummy_output
