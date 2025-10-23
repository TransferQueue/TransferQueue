# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import inspect
from typing import Callable

from viztracer import VizTracer


class VizTracerProfiler:
    """
    ...
    """
    def __init__(self, **kwargs):
        self.trace = VizTracer(**kwargs)
        self.this_step = False

    def start(self, **kwargs):
        if not self.this_step:
            self.trace.start()
            self.this_step = True
    
    def stop(self, **kwargs):
        if self.this_step:
            self.trace.stop()
            self.trace.save()
            self.this_step = False

    @classmethod
    def trace(cls) -> Callable:
        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                profiler = getattr(self, "profiler", None)
                if not profiler:
                    return func(self, *args, **kwargs)

                profiler.start()
                try:
                    return func(self, *args, **kwargs)
                finally:
                    profiler.stop()
            
            @functools.wraps(func)
            async def async_wrapper(self, *args, **kwargs):
                profiler = getattr(self, "profiler", None)
                if not profiler:
                    return await func(self, *args, **kwargs)
                
                profiler.start()
                try:
                    return await func(self, *args, **kwargs)
                finally:
                    profiler.stop()

            return async_wrapper if inspect.iscoroutinefunction(func) else wrapper

        return decorator


class ProfilerExtension:
    def __init__(self, profiler: VizTracerProfiler):
        self.profiler = profiler
