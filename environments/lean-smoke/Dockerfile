FROM mcr.microsoft.com/dotnet/sdk:10.0@sha256:de4fe90a98845d75910b9769662465c84de1a934665c431279d1382b2bddc02b AS compiler
WORKDIR /source
COPY . .
RUN dotnet build Launcher/QuantConnect.Lean.Launcher.csproj -c Release --nologo -v minimal -p:RunAnalyzers=false -p:EnableNETAnalyzers=false
FROM quantagentlab:smoke-a233bb5
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libicu72 libunwind8 && rm -rf /var/lib/apt/lists/*
COPY --from=compiler /usr/share/dotnet /usr/share/dotnet
COPY --from=compiler /source/Launcher/bin/Release /Lean/Launcher
COPY --from=compiler /source/Data /Lean/Data
COPY --from=compiler /source/Common/AlgorithmImports.py /Lean/Launcher/AlgorithmImports.py
ENV DOTNET_ROOT=/usr/share/dotnet PYTHONNET_PYDLL=/usr/local/lib/libpython3.11.so PYTHONPATH=/app:/Lean/Launcher DOTNET_CLI_TELEMETRY_OPTOUT=1
USER appuser
WORKDIR /Lean/Launcher
ENTRYPOINT ["/usr/share/dotnet/dotnet", "QuantConnect.Lean.Launcher.dll"]
