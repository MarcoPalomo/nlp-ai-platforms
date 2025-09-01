{{- define "torchserve.name" -}}
torchserve
{{- end }}

{{- define "torchserve.fullname" -}}
{{ .Release.Name }}-torchserve
{{- end }}