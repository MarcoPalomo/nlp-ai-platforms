{{- define "mistral-server.fullname" -}}
{{- printf "%s-%s" .Release.Name "mistral-server" | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{- define "mistral-server.labels" -}}
app.kubernetes.io/name: mistral-server
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
component: llm
model: {{ .Values.model.name | default "mistral" | quote }}
{{- end }}
