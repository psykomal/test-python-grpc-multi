FROM python:3.9.14-buster


# ARG NONROOT_USER
# RUN echo "User will be $NONROOT_USER"
# ENV PYTHON_USER=$NONROOT_USER

# Create unprivileged user with a home dir and using bash
# RUN useradd -ms /bin/bash $PYTHON_USER

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


COPY --chmod=0755 ./entrypoint.sh ./entrypoint.sh
# COPY --chown=$PYTHON_USER:$PYTHON_USER --chmod=0755 ./post-initialization.sh ./post-initialization.sh
# # If you have a requirements.txt for the project, uncomment this and
# # adjust post-initialization.sh to use it
# COPY --chown=$PYTHON_USER:$PYTHON_USER requirements.txt .
COPY --chmod=0755 ./post-initialization.sh ./post-initialization.sh
# If you have a requirements.txt for the project, uncomment this and
# adjust post-initialization.sh to use it
COPY requirements.txt .

COPY . ./app

EXPOSE 50051

# ENTRYPOINT ["./docker-entrypoint.sh"]

# CMD ["sleep", "inf"]
# CMD ["/bin/bash", "-c", "./entrypoint.sh $PYTHON_USER"]
CMD ["/bin/bash", "-c", "./entrypoint.sh"]
