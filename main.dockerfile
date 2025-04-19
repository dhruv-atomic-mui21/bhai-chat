FROM alpine:3.17

# Install necessary packages (if any)
RUN apk add --no-cache shadow

# Create group and user with specific UID/GID
RUN groupadd -g 1001 appgroup \
 && useradd -u 1001 -g appgroup -m -s /bin/sh appuser

# Ensure ownership of working directory
WORKDIR /app
COPY . /app
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Default command
CMD ["./start.sh"]
