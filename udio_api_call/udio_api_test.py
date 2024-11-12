from ..UdioWrapper.udio_wrapper.__init__ import UdioWrapper

auth_token = "base64-eyJhY2Nlc3NfdG9rZW4iOiJleUpoYkdjaU9pSklVekkxTmlJc0ltdHBaQ0k2SWxKSFZrdG9Wek5OY1NzeVZ6aHhjRGtpTENKMGVYQWlPaUpLVjFRaWZRLmV5SmhZV3dpT2lKaFlXd3hJaXdpWVcxeUlqcGJleUp0WlhSb2IyUWlPaUp2WVhWMGFDSXNJblJwYldWemRHRnRjQ0k2TVRjeU9UWXhPRFUxTkgxZExDSmhjSEJmYldWMFlXUmhkR0VpT25zaWNISnZkbWxrWlhJaU9pSm5iMjluYkdVaUxDSndjbTkyYVdSbGNuTWlPbHNpWjI5dloyeGxJbDE5TENKaGRXUWlPaUpoZFhSb1pXNTBhV05oZEdWa0lpd2laVzFoYVd3aU9pSm9ZWFpoY21ScWRtUkFaMjFoYVd3dVkyOXRJaXdpWlhod0lqb3hOek13TWpNd01qUTNMQ0pwWVhRaU9qRTNNekF5TWpZMk5EY3NJbWx6WDJGdWIyNTViVzkxY3lJNlptRnNjMlVzSW1semN5STZJbWgwZEhCek9pOHZiV1p0Y0hocVpXMWhZM05vWm1Od2VtOXpiSFV1YzNWd1lXSmhjMlV1WTI4dllYVjBhQzkyTVNJc0luQm9iMjVsSWpvaUlpd2ljbTlzWlNJNkltRjFkR2hsYm5ScFkyRjBaV1FpTENKelpYTnphVzl1WDJsa0lqb2lOMkl6TlRkak1qQXRPREJoTWkwMFptUXhMV0UxWVRndFlUTmpaR015WWpSbVkyWTVJaXdpYzNWaUlqb2lPVGMwTmprelltSXRNRGxtT1MwMFptWTVMV0k0WkRndFpEZzJOemxtTW1ZM09XWmhJaXdpZFhObGNsOXRaWFJoWkdGMFlTSTZleUpoZG1GMFlYSmZkWEpzSWpvaWFIUjBjSE02THk5c2FETXVaMjl2WjJ4bGRYTmxjbU52Ym5SbGJuUXVZMjl0TDJFdlFVTm5PRzlqVEd3MVdERmZOa001U0dGWWJtbHlUVW8yTkVWSmFqQlJWamd4TTFsTVdHbEllVEYyZUUxcE5VTkdTM1J3VVdwRlVUMXpPVFl0WXlJc0ltVnRZV2xzSWpvaWFHRjJZWEprYW5aa1FHZHRZV2xzTG1OdmJTSXNJbVZ0WVdsc1gzWmxjbWxtYVdWa0lqcDBjblZsTENKbWRXeHNYMjVoYldVaU9pSkl3NlYyWVhKa0lFUmhiR1Z1WnlJc0ltbHpjeUk2SW1oMGRIQnpPaTh2WVdOamIzVnVkSE11WjI5dloyeGxMbU52YlNJc0ltNWhiV1VpT2lKSXc2VjJZWEprSUVSaGJHVnVaeUlzSW01bFpXUnpYMjl1WW05aGNtUnBibWNpT21aaGJITmxMQ0p1WlhkZmRYTmxjaUk2Wm1Gc2MyVXNJbkJvYjI1bFgzWmxjbWxtYVdWa0lqcG1ZV3h6WlN3aWNHbGpkSFZ5WlNJNkltaDBkSEJ6T2k4dmJHZ3pMbWR2YjJkc1pYVnpaWEpqYjI1MFpXNTBMbU52YlM5aEwwRkRaemh2WTB4c05WZ3hYelpET1VoaFdHNXBjazFLTmpSRlNXb3dVVlk0TVROWlRGaHBTSGt4ZG5oTmFUVkRSa3QwY0ZGcVJWRTljemsyTFdNaUxDSndjbTkyYVdSbGNsOXBaQ0k2SWpFd05EVTJNVEU1TkRBM05EZzFORFE0T0RreU9TSXNJbk4xWWlJNklqRXdORFUyTVRFNU5EQTNORGcxTkRRNE9Ea3lPU0o5TENKMWMyVnlYM0p2YkdVaU9tNTFiR3g5Lm9qR1RRQXpZV05UQmRIMENYdWNod3kxU3VFSlh0MUM3UEZWVFhlMk9iVTgiLCJ0b2tlbl90eXBlIjoiYmVhcmVyIiwiZXhwaXJlc19pbiI6MzYwMCwiZXhwaXJlc19hdCI6MTczMDIzMDI0NywicmVmcmVzaF90b2tlbiI6IkxkcjFZbElHUFlLVFhValpqRjVKWFEiLCJ1c2VyIjp7ImlkIjoiOTc0NjkzYmItMDlmOS00ZmY5LWI4ZDgtZDg2NzlmMmY3OWZhIiwiYXVkIjoiYXV0aGVudGljYXRlZCIsInJvbGUiOiJhdXRoZW50aWNhdGVkIiwiZW1haWwiOiJoYXZhcmRqdmRAZ21haWwuY29tIiwiZW1haWxfY29uZmlybWVkX2F0IjoiMjAyNC0wNS0wOFQxMTozNDozOS4zNDE3MzRaIiwicGhvbmUiOiIiLCJjb25maXJtZWRfYXQiOiIyMDI0LTA1LTA4VDExOjM0OjM5LjM0MTczNFoiLCJsYXN0X3NpZ25faW5fYXQiOiIyMDI0LTEwLTIyVDE3OjM1OjU0Ljg5MTU2M1oiLCJhcHBfbWV0YWRhdGEiOnsicHJvdmlkZXIiOiJnb29nbGUiLCJwcm92aWRlcnMiOlsiZ29vZ2xlIl19LCJ1c2VyX21ldGFkYXRhIjp7ImF2YXRhcl91cmwiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NMbDVYMV82QzlIYVhuaXJNSjY0RUlqMFFWODEzWUxYaUh5MXZ4TWk1Q0ZLdHBRakVRPXM5Ni1jIiwiZW1haWwiOiJoYXZhcmRqdmRAZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsImZ1bGxfbmFtZSI6IkjDpXZhcmQgRGFsZW5nIiwiaXNzIjoiaHR0cHM6Ly9hY2NvdW50cy5nb29nbGUuY29tIiwibmFtZSI6IkjDpXZhcmQgRGFsZW5nIiwibmVlZHNfb25ib2FyZGluZyI6ZmFsc2UsIm5ld191c2VyIjpmYWxzZSwicGhvbmVfdmVyaWZpZWQiOmZhbHNlLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUNnOG9jTGw1WDFfNkM5SGFYbmlyTUo2NEVJajBRVjgxM1lMWGlIeTF2eE1pNUNGS3RwUWpFUT1zOTYtYyIsInByb3ZpZGVyX2lkIjoiMTA0NTYxMTk0MDc0ODU0NDg4OTI5Iiwic3ViIjoiMTA0NTYxMTk0MDc0ODU0NDg4OTI5In0sImlkZW50aXRpZXMiOlt7I"  # Replace this with your actual authentication token
udio_wrapper = UdioWrapper(auth_token)

