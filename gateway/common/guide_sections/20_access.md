## Access

**Base URL:** `{{base_url}}` for this response -- use whatever host and port you actually reach this
API on; there is no other fixed public URL documented here.

**Network reachability:** this API has no CORS configuration, so it isn't meant to be called
directly from a browser running on a different origin. It's built for server-to-server integration
(your backend calling this one directly), typically over a private network or a forwarded port --
it is not exposed on the open internet by default.

**Authentication:** none. This API does not use authentication -- send requests with no
`Authorization` header at all.