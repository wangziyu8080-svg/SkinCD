# SkinCD Data Preparation Guide

This guide lists the public dataset sources used by the project and illustrates a portable local layout. Review and comply with each dataset's license and access terms.


## Canonical Data Root

Choose a local data root:

        export SKINCD_DATA_ROOT=/path/to/skincd-data

Pass this root to experiment scripts through variables such as `HAM10K_IMAGE_ROOT`.

## Dataset Order (matching skindata directory)

| Dataset | Modality | Official Source |
| --- | --- | --- |
| 2 BCN20000 | Dermoscopy | https://api.isic-archive.com/collections/249/ |
| 1 DDI | Clinical | https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965 |
| 1 Dermnet | Clinical | https://www.kaggle.com/datasets/shubhamgoel27/dermnet |
| Form Fitzpatrick17k | Clinical | https://github.com/mattgroh/fitzpatrick17k |
| 1 HAM10K | Dermoscopy | https://challenge.isic-archive.com/data/#2018 |
| HIBA | Dermoscopy | https://api.isic-archive.com/collections/175/ |
| 2 ISIC2019 | Dermoscopy | https://api.isic-archive.com/collections/65/ |
| 1 MM-Skin | Clinical, Dermoscopy, Pathology | https://drive.google.com/drive/folders/1gAQOpJjzldpqegIcZcKX5_2Wau54taJ4?usp=sharing |
| MSKCC | Dermoscopy | https://api.isic-archive.com/collections/163/ |
| 1 PAD | Clinical | https://data.mendeley.com/datasets/zr7vgbcyr2/1 |
| 1 Patch16 | Pathology | https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/7QCR8S |
| 1 SCIN | Clinical | https://console.cloud.google.com/storage/browser/dx-scin-public-data?inv=1&invt=Abw9Eg |

## BCN20000

Sources (official):

- Collection page: https://api.isic-archive.com/collections/249/
- Metadata direct CSV: https://api.isic-archive.com/collections/249/metadata/
- Collection ZIP API (returns temporary download URL): https://api.isic-archive.com/api/v2/zip-download/url/

Verified download commands:

        mkdir -p "${SKINCD_DATA_ROOT}/BCN20000"
        cd "${SKINCD_DATA_ROOT}/BCN20000"

        # 1) Metadata CSV
        curl -L -o bcn20000_metadata_download.csv \
            https://api.isic-archive.com/collections/249/metadata/

        # 2) Collection ZIP (API returns a JSON string, not an object)
        ZIP_URL=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d '{"collections": "249"}' \
            https://api.isic-archive.com/api/v2/zip-download/url/ | tr -d '"')

        # 3) Download with resume support
        wget -c "$ZIP_URL" -O bcn20000_collection_download.zip

## DDI

Source:

- https://stanfordaimi.azurewebsites.net/datasets/35866158-8196-48d8-87bf-50dca81df965

## Dermnet

Source:

- https://www.kaggle.com/datasets/shubhamgoel27/dermnet

Example download command:

        mkdir -p Dermnet
        python - <<'PY'
        import pathlib
        import shutil
        import kagglehub

        src = pathlib.Path(kagglehub.dataset_download('shubhamgoel27/dermnet'))
        dst = pathlib.Path('Dermnet')
        dst.mkdir(parents=True, exist_ok=True)

        for p in src.iterdir():
            target = dst / p.name
            if p.is_dir():
                shutil.copytree(p, target, dirs_exist_ok=True)
            else:
                shutil.copy2(p, target)
        PY

## Fitzpatrick17k

Source:

- https://github.com/mattgroh/fitzpatrick17k

## HAM10K

Source:

- https://challenge.isic-archive.com/data/#2018

## HIBA

Source:

- https://api.isic-archive.com/collections/175/

## ISIC2019

Sources (official):

- Collection page: https://api.isic-archive.com/collections/65/
- Metadata direct CSV: https://api.isic-archive.com/collections/65/metadata/
- Collection ZIP API (returns temporary download URL): https://api.isic-archive.com/api/v2/zip-download/url/

Verified download commands:

        mkdir -p "${SKINCD_DATA_ROOT}/ISIC2019"
        cd "${SKINCD_DATA_ROOT}/ISIC2019"

        # 1) Metadata CSV
        curl -L -o isic2019_metadata_download.csv \
            https://api.isic-archive.com/collections/65/metadata/

        # 2) Collection ZIP (API returns a JSON string, not an object)
        ZIP_URL=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d '{"collections": "65"}' \
            https://api.isic-archive.com/api/v2/zip-download/url/ | tr -d '"')

        # 3) Download with resume support
        wget -c "$ZIP_URL" -O isic2019_collection_download.zip

Validation:

        ls -lh isic2019_metadata_download.csv isic2019_collection_download.zip
        unzip -t isic2019_collection_download.zip

## MM-Skin

Source:

- https://drive.google.com/drive/folders/1gAQOpJjzldpqegIcZcKX5_2Wau54taJ4?usp=sharing

## MSKCC

Source:

- https://api.isic-archive.com/collections/163/

## PAD

Source:

- https://data.mendeley.com/datasets/zr7vgbcyr2/1

Example download commands:

        mkdir -p "${SKINCD_DATA_ROOT}/PAD/raw" "${SKINCD_DATA_ROOT}/PAD/images"

        cat > /tmp/pad_mendeley_files.txt <<'EOF'
        imgs_part_1.zip|1245184680|0ab44f60938bf57445e12f518a8878954cc734e6b0aec6d01194e2d26b4b2dca|https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/1cc2f71f-20a2-412d-b746-a9b9bc20c966/file_downloaded
        imgs_part_2.zip|1126646990|e2d9a3cbd58e823f5ae33163c48643e7d1b54ae3f9e145f01f8e9f16a363a60b|https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/559a60ed-5504-475d-996c-6a8bc253b5e7/file_downloaded
        imgs_part_3.zip|1220565093|ecc4ef10143a43e1d01cb736773148607a78b530417eb76f2c38ad24bf5d0d2c|https://data.mendeley.com/public-files/datasets/zr7vgbcyr2/files/34dcdf8e-e5f1-4b35-aa0b-5135051ff852/file_downloaded
        EOF

        while IFS='|' read -r name expect_size expect_sha url; do
            out="${SKINCD_DATA_ROOT}/PAD/raw/$name"
            tmp="${out}.part"
            rm -f "$tmp"
            wget --tries=0 --retry-connrefused --waitretry=5 --read-timeout=30 --timeout=30 -O "$tmp" "$url"
            [ "$(stat -c%s "$tmp")" -eq "$expect_size" ]
            echo "$expect_sha  $tmp" | sha256sum -c -
            mv -f "$tmp" "$out"
        done < /tmp/pad_mendeley_files.txt

        for z in "${SKINCD_DATA_ROOT}"/PAD/raw/imgs_part_1.zip "${SKINCD_DATA_ROOT}"/PAD/raw/imgs_part_2.zip "${SKINCD_DATA_ROOT}"/PAD/raw/imgs_part_3.zip; do
            unzip -q -o "$z" -d "${SKINCD_DATA_ROOT}/PAD/images"
        done

## Patch16

Source:

- https://heidata.uni-heidelberg.de/dataset.xhtml?persistentId=doi:10.11588/data/7QCR8S

## SCIN

Source:

- https://console.cloud.google.com/storage/browser/dx-scin-public-data?inv=1&invt=Abw9Eg

Re-download command (full bucket sync):

        mkdir -p SCIN
        gsutil -m rsync -r gs://dx-scin-public-data SCIN

If gsutil is unavailable, use Google Storage JSON API fallback:

        mkdir -p SCIN
        python - <<'PY'
        import json
        import pathlib
        import urllib.parse
        import urllib.request

        out = pathlib.Path('SCIN')
        out.mkdir(parents=True, exist_ok=True)

        token = None
        while True:
            base = 'https://storage.googleapis.com/storage/v1/b/dx-scin-public-data/o?maxResults=1000'
            if token:
                base += '&pageToken=' + urllib.parse.quote(token, safe='')
            with urllib.request.urlopen(base) as r:
                obj = json.load(r)

            for item in obj.get('items', []):
                name = item['name']
                media = item['mediaLink']
                dst = out / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(media, dst)

            token = obj.get('nextPageToken')
            if not token:
                break
        PY

 
 
