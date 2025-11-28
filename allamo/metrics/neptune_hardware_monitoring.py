"""
Monitors system and process metrics in a background thread and logs them to Neptune.

Source: https://github.com/neptune-ai/scale-examples/tree/main/utils/monitoring_tools/hardware_monitoring
"""
import atexit
import contextlib
import os
import socket
import threading
import time
import traceback
from typing import Any, Dict, Optional

import psutil
from neptune_scale import Run
from allamo.logging import logger

__version__ = "0.3.0"

try:
    import torch

    _torch_available = True
except ImportError:
    _torch_available = False

try:
    import pynvml

    _pynvml_available = True

except ImportError:
    logger.warning(
        "`pynvml` is not installed. GPU monitoring will be disabled. Install using `pip install nvidia-ml-py` if you want GPU metrics."
    )
    _pynvml_available = False


class SystemMetricsMonitor:
    """
    Monitors system and process metrics in a background thread and logs them to Neptune.

    This class collects metrics such as CPU, memory, disk, network, GPU (if available), and process resource usage.
    Metrics are logged at a configurable sampling rate to a Neptune run, under a specified namespace.
    The monitor is robust to partial hardware failures and can be used as a context manager.
    """

    def __init__(
        self,
        run: Run,
        sampling_rate: float = 5.0,
        namespace: str = "runtime",
    ) -> None:
        """
        Initialize the SystemMetricsMonitor.

        Args:
            run (Run): Neptune Run object for logging metrics.
            sampling_rate (float, optional): How often to sample metrics (in seconds). Default is 5.0.
            namespace (str, optional): Namespace where the metrics will be logged in the Neptune run. Default is "runtime".
        """
        self.run = run
        self.namespace = namespace
        self.sampling_rate = sampling_rate
        self._stop_event = threading.Event()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._proc = psutil.Process(os.getpid())
        # Handle _fork_step which a bound method for dummy experiments on non-zero ranks
        if callable(self.run._fork_step):
            _fork_step = self.run._fork_step()
        else:
            _fork_step = self.run._fork_step
        self._monitoring_step = _fork_step + 1 if _fork_step is not None else 0
        # Last time stamp GPU SM process information was retrieved
        self._last_process_time_stamp = 0
        self.hostname = socket.gethostname()

        # Prime psutil.cpu_percent to avoid initial 0.0 reading
        psutil.cpu_percent()

        if _pynvml_available:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.has_gpu = True

                # Register NVML shutdown at exit
                atexit.register(pynvml.nvmlShutdown)
            except Exception:
                logger.warning(
                    "No NVIDIA GPU available or driver issues. GPU monitoring will be disabled."
                )
                self.has_gpu = False
                self.gpu_count = 0
        else:
            self.has_gpu = False
            self.gpu_count = 0

        self._log_system_details()

    def _log_system_details(self) -> None:
        """
        Log static system details (device, CPU, GPU, hostname) as Neptune run configs.
        Also logs GPU names if available.
        """
        system_details = {
            "gpu_num": self.gpu_count,
            "cpu_num": psutil.cpu_count(logical=False),
            "cpu_logical_num": psutil.cpu_count(logical=True),
            "hostname": self.hostname,
        }

        if _torch_available:
            system_details["device"] = str(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

        self.run.log_configs(
            {f"{self.namespace}/details/{key}": value for key, value in system_details.items()}
        )

        if self.has_gpu and _pynvml_available:
            try:
                for i in range(self.gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle)
                    self.run.log_configs({f"{self.namespace}/details/gpu/{i}/name": name})
            except Exception as e:
                logger.warning(f"Error getting GPU details on {self.hostname}: {e}")

    def _collect_metrics(self) -> Dict[str, Any]:
        """
        Collect current system and process metrics.

        Returns:
            dict: Dictionary of collected metrics, keyed by metric name.
        """
        prefix = f"{self.namespace}/monitoring"
        metrics = {}
        self._collect_cpu_metrics(metrics, prefix)
        self._collect_memory_metrics(metrics, prefix)
        self._collect_disk_metrics(metrics, prefix)
        self._collect_network_metrics(metrics, prefix)
        self._collect_gpu_metrics(metrics, prefix)
        self._collect_process_metrics(metrics, prefix)
        return metrics

    def _collect_cpu_metrics(self, metrics: Dict[str, Any], prefix: str) -> None:
        """
        Collect CPU usage metrics and add them to the metrics dictionary.

        Args:
            metrics (dict): The metrics dictionary to update.
            prefix (str): The namespace prefix for metric keys.
        """
        metrics[f"{prefix}/cpu/percent"] = psutil.cpu_percent()

    def _collect_memory_metrics(self, metrics: Dict[str, Any], prefix: str) -> None:
        """
        Collect memory and swap usage metrics and add them to the metrics dictionary.

        Args:
            metrics (dict): The metrics dictionary to update.
            prefix (str): The namespace prefix for metric keys.
        """
        virtual_memory = psutil.virtual_memory()
        metrics[f"{prefix}/memory/virtual_used_GiB"] = virtual_memory.used / (1024**3)
        metrics[f"{prefix}/memory/virtual_utilized_percent"] = virtual_memory.percent
        swap_memory = psutil.swap_memory()
        metrics[f"{prefix}/memory/swap_used_MiB"] = swap_memory.used / (1024**2)
        metrics[f"{prefix}/memory/swap_utilized_percent"] = swap_memory.percent

    def _collect_disk_metrics(self, metrics: Dict[str, Any], prefix: str) -> None:
        """
        Collect disk I/O metrics and add them to the metrics dictionary.

        Args:
            metrics (dict): The metrics dictionary to update.
            prefix (str): The namespace prefix for metric keys.
        """
        disk_io = psutil.disk_io_counters()

        if hasattr(self, "_previous_disk_io"):
            # Calculate deltas for disk I/O metrics
            metrics[f"{prefix}/disk/read_count"] = (
                disk_io.read_count - self._previous_disk_io.read_count
            )
            metrics[f"{prefix}/disk/write_count"] = (
                disk_io.write_count - self._previous_disk_io.write_count
            )
            metrics[f"{prefix}/disk/read_MiB"] = (
                disk_io.read_bytes - self._previous_disk_io.read_bytes
            ) / (1024**2)
            metrics[f"{prefix}/disk/write_MiB"] = (
                disk_io.write_bytes - self._previous_disk_io.write_bytes
            ) / (1024**2)
        else:
            # Initialize metrics with zero deltas for the first interval
            metrics[f"{prefix}/disk/read_count"] = 0
            metrics[f"{prefix}/disk/write_count"] = 0
            metrics[f"{prefix}/disk/read_MiB"] = 0
            metrics[f"{prefix}/disk/write_MiB"] = 0

        # Update previous disk I/O counters
        self._previous_disk_io = disk_io

    def _collect_network_metrics(self, metrics: Dict[str, Any], prefix: str) -> None:
        """
        Collect network I/O metrics and add them to the metrics dictionary.

        Args:
            metrics (dict): The metrics dictionary to update.
            prefix (str): The namespace prefix for metric keys.
        """
        network_io = psutil.net_io_counters()

        if hasattr(self, "_previous_network_io"):
            metrics[f"{prefix}/network/sent_MiB"] = (
                network_io.bytes_sent - self._previous_network_io.bytes_sent
            ) / (1024**2)
            metrics[f"{prefix}/network/recv_MiB"] = (
                network_io.bytes_recv - self._previous_network_io.bytes_recv
            ) / (1024**2)
        else:
            metrics[f"{prefix}/network/sent_MiB"] = 0
            metrics[f"{prefix}/network/recv_MiB"] = 0

        self._previous_network_io = network_io

    def _collect_gpu_metrics(self, metrics: Dict[str, Any], prefix: str) -> None:
        """
        Collect GPU metrics for all available GPUs and add them to the metrics dictionary.
        Handles errors on a per-GPU and per-metric basis to ensure partial failures do not prevent collection from other GPUs.

        Args:
            metrics (dict): The metrics dictionary to update.
            prefix (str): The namespace prefix for metric keys.
        """
        if not self.has_gpu or not _pynvml_available:
            return
        for i in range(self.gpu_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            except Exception as e:
                logger.warning(f"Error getting handle for GPU {i} on {self.hostname}: {e}")
                continue
            # Memory info
            try:
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                metrics[f"{prefix}/gpu/{i}/memory_used_MiB"] = memory.used / (1024**2)
                metrics[f"{prefix}/gpu/{i}/memory_total_MiB"] = memory.total / (1024**2)
                metrics[f"{prefix}/gpu/{i}/memory_free_MiB"] = memory.free / (1024**2)
                metrics[f"{prefix}/gpu/{i}/memory_utilized_percent"] = (
                    (memory.used / memory.total) * 100 if memory.total else 0.0
                )
            except Exception as e:
                logger.warning(f"Error getting memory info for GPU {i} on {self.hostname}: {e}")
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                metrics[f"{prefix}/gpu/{i}/temperature_celsius"] = temp
            except Exception as e:
                logger.warning(f"Error getting temperature for GPU {i} on {self.hostname}: {e}")
            # Utilization
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                metrics[f"{prefix}/gpu/{i}/gpu_utilization_percent"] = utilization.gpu
            except Exception as e:
                logger.warning(f"Error getting utilization for GPU {i} on {self.hostname}: {e}")
            # Power usage (if available)
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
                metrics[f"{prefix}/gpu/{i}/power_usage_watts"] = power
            except Exception as e:
                logger.warning(f"Error getting power usage for GPU {i} on {self.hostname}: {e}")
            # SM Process Utilization
            try:
                # SM Utilization samples is returned as a list of samples
                # There are as many as samples as active processes in the time stamp interval
                # For more details, see
                # https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries_1gb0ea5236f5e69e63bf53684a11c233bd
                sm_utilization_samples: list[pynvml.c_nvmlProcessUtilizationSample_t] = (
                    pynvml.nvmlDeviceGetProcessUtilization(handle, self._last_process_time_stamp)
                )
            except pynvml.nvmlExceptionClass(pynvml.NVML_ERROR_NOT_FOUND) as e:
                # If no valid sample entries are found since the last seen time stamp, NVML_ERROR_NOT_FOUND is returned.
                # It is expected if no process is active during two consecutive calls.
                sm_util = 0
                mem_util = 0
            except Exception as e:
                logger.warning(
                    f"Error getting process utilization for GPU {i} on {self.hostname}: {e}"
                )
                sm_util = None
                mem_util = None
            else:
                # We assume process utilization is given in percentage and can be summed
                # We expect only one process to be active at a time during training
                sm_util = sum(sample.smUtil for sample in sm_utilization_samples)
                mem_util = sum(sample.memUtil for sample in sm_utilization_samples)
                self._last_process_time_stamp = max(
                    sample.timeStamp for sample in sm_utilization_samples
                )
            finally:
                if sm_util is not None and mem_util is not None:
                    metrics[f"{prefix}/gpu/{i}/sm_utilization_percent"] = sm_util
                    metrics[f"{prefix}/gpu/{i}/sm_memory_utilization_percent"] = mem_util

    def _collect_process_metrics(self, metrics: Dict[str, Any], prefix: str) -> None:
        """
        Collect resource usage metrics for the current Python process (memory, threads, file descriptors).

        Args:
            metrics (dict): The metrics dictionary to update.
            prefix (str): The namespace prefix for metric keys.
        """
        mem_info = self._proc.memory_info()
        metrics[f"{prefix}/process/rss_memory_MiB"] = mem_info.rss / (1024**2)
        metrics[f"{prefix}/process/vms_memory_GiB"] = mem_info.vms / (1024**3)
        metrics[f"{prefix}/process/num_threads"] = self._proc.num_threads()

        # Open file descriptors (Unix)
        if hasattr(self._proc, "num_fds"):
            metrics[f"{prefix}/process/num_fds"] = self._proc.num_fds()
        # Open handles (Windows)
        elif hasattr(self._proc, "num_handles"):
            metrics[f"{prefix}/process/num_fds"] = self._proc.num_handles()

    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop that runs in a background thread.
        Collects and logs metrics at the configured sampling rate.
        Handles and logs errors during metric collection.
        """
        while not self._stop_event.is_set():
            start_time = time.monotonic()

            try:
                metrics = self._collect_metrics()
                self.run.log_metrics(data=metrics, step=self._monitoring_step)
                self._monitoring_step += 1

            except Exception as e:
                logger.warning(f"Error collecting metrics: {e}\n{traceback.format_exc()}")

            finally:
                elapsed = time.monotonic() - start_time
                if elapsed > self.sampling_rate:
                    logger.debug(
                        f"Metric collection took {elapsed:.2f}s which exceeds the sampling rate of {self.sampling_rate}s."
                    )
                sleep_time = max(0, self.sampling_rate - elapsed)
                time.sleep(sleep_time)

    def start(self) -> None:
        """
        Start the monitoring thread if not already running.
        """
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_event.clear()
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()

    def stop(self) -> None:
        """
        Stop the monitoring thread and wait for it to finish.
        Also shuts down NVML if GPU monitoring was enabled.
        """
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._stop_event.set()
            self._monitoring_thread.join(timeout=self.sampling_rate + 2)
            if self._monitoring_thread.is_alive():
                logger.debug(
                    "Monitoring thread did not terminate after join(). Potential hang-up detected."
                )
            self._monitoring_thread = None

            if self.has_gpu and _pynvml_available:
                with contextlib.suppress(Exception):
                    pynvml.nvmlShutdown()

    def __enter__(self) -> "SystemMetricsMonitor":
        """
        Enter the context manager, starting the monitoring thread.

        Returns:
            SystemMetricsMonitor: The monitor instance.
        """
        self.start()
        return self

    def __exit__(
        self, exc_type: Optional[type], exc_val: Optional[BaseException], exc_tb: Optional[Any]
    ) -> None:
        """
        Exit the context manager, ensuring monitoring is stopped.
        """
        self.stop()